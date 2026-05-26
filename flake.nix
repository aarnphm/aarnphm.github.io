{
  description = "aarnphm's garden";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    git-hooks,
    nixpkgs,
  }: let
    inherit (nixpkgs) lib;
    fs = lib.fileset;
    systems = [
      "aarch64-darwin"
      "aarch64-linux"
      "x86_64-darwin"
      "x86_64-linux"
    ];
    forAllSystems = lib.genAttrs systems;
    package = builtins.fromJSON (builtins.readFile ./package.json);
    inherit (package) version;
    gitRev = self.rev or self.dirtyRev or "dirty";
    gardenFiles = fs.unions [
      (./. + "/@types")
      ./content
      ./migrations
      ./quartz
      ./worker
      ./.node-version
      ./.lfsconfig
      ./.npmrc
      ./.oxfmtrc.json
      ./.pnpmfile.cjs
      ./drizzle.config.ts
      ./globals.d.ts
      ./index.d.ts
      ./package.json
      ./pnpm-lock.yaml
      ./pnpm-workspace.yaml
      ./pyproject.toml
      ./quartz.config.ts
      ./quartz.layout.ts
      ./tsconfig.json
      ./uv.lock
      ./worker-configuration.d.ts
      ./wrangler.toml
    ];
    gardenSrc = fs.toSource {
      root = ./.;
      fileset = gardenFiles;
    };
    vaultSrc = fs.toSource {
      root = ./.;
      fileset = ./content;
    };
    mkPackages = system: let
      pkgs = import nixpkgs {inherit system;};
      nodejs = pkgs.nodejs_24;
      pnpm = pkgs.pnpm_10.override {inherit nodejs;};
      python = pkgs.python313;
      fetchPnpmInstallFlags = [
        "--frozen-lockfile"
        "--strict-peer-dependencies"
      ];
      setupBuildEnv = ''
        export HOME="$TMPDIR/home"
        mkdir -p "$HOME"
        export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
        export npm_config_cache="$TMPDIR/npm-cache"
        export UV_CACHE_DIR="$TMPDIR/uv-cache"
        pnpmInstallFlags=(
          "--frozen-lockfile"
          "--strict-peer-dependencies"
        )
      '';
      pnpmDeps = pkgs.fetchPnpmDeps {
        pname = "garden-pnpm-deps";
        inherit version;
        src = gardenSrc;
        inherit pnpm;
        pnpmInstallFlags = fetchPnpmInstallFlags;
        fetcherVersion = 3;
        hash = "sha256-tk6MRfVmf/v+NW1nbrHkwTslNPlSJbz3bxhu6+7ANco=";
      };
      quartz = pkgs.stdenvNoCC.mkDerivation {
        pname = "garden-quartz";
        inherit version;
        src = gardenSrc;
        nativeBuildInputs = [
          nodejs
          pnpm
          pkgs.pnpmConfigHook
        ];
        inherit pnpmDeps;
        preConfigure = setupBuildEnv;
        dontBuild = true;
        dontFixup = true;
        installPhase = ''
          runHook preInstall
          mkdir -p "$out/share/garden"
          cp -a . "$out/share/garden"
          substituteInPlace "$out/share/garden"/node_modules/.bin/* --replace-quiet "$PWD" "$out/share/garden"
          runHook postInstall
        '';
      };
      vault = pkgs.stdenvNoCC.mkDerivation {
        pname = "garden-vault";
        inherit version;
        src = vaultSrc;
        dontBuild = true;
        dontFixup = true;
        installPhase = ''
          runHook preInstall
          mkdir -p "$out"
          cp -a content/. "$out"
          runHook postInstall
        '';
      };
      site = pkgs.stdenvNoCC.mkDerivation {
        pname = "garden-site";
        inherit version;
        src = gardenSrc;
        nativeBuildInputs = [
          nodejs
          pnpm
          pkgs.fd
          pkgs.gitMinimal
          pkgs.pnpmConfigHook
          pkgs.uv
          python
        ];
        inherit pnpmDeps;
        preConfigure = setupBuildEnv;
        dontFixup = true;
        buildPhase = ''
          runHook preBuild
          export NODE_ENV=production
          export GITHUB_SHA="${gitRev}"
          pnpm exec quartz/bootstrap-cli.mjs build --concurrency 10 --bundleInfo --verbose --output public
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          mkdir -p "$out"
          cp -a public/. "$out"
          runHook postInstall
        '';
        passthru = {
          inherit quartz vault;
        };
      };
      deployable =
        pkgs.runCommand "garden-deployable-${version}" {
          nativeBuildInputs = [pkgs.fd];
          dontFixup = true;
        } ''
          mkdir -p "$out"
          cp -a ${site}/. "$out"
          fd --glob "*.ddl" "$out" -x rm
          fd --glob "*.war" "$out" -x rm
          rm -f "$out/embeddings-text.jsonl"
        '';
    in {
      default = deployable;
      inherit deployable pnpmDeps quartz site vault;
    };
    mkApps = system: let
      pkgs = import nixpkgs {inherit system;};
      nodejs = pkgs.nodejs_24;
      pnpm = pkgs.pnpm_10.override {inherit nodejs;};
      app = drv: {
        type = "app";
        program = "${drv}/bin/${drv.name}";
      };
      build = pkgs.writeShellApplication {
        name = "garden-build";
        runtimeInputs = [
          pkgs.gitMinimal
          pkgs.nix
        ];
        text = ''
          root="''${GARDEN_ROOT:-}"
          if [ -z "$root" ]; then
            root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          fi
          exec nix build "$root#deployable" "$@"
        '';
      };
      buildSite = pkgs.writeShellApplication {
        name = "garden-build-site";
        runtimeInputs = [
          pkgs.gitMinimal
          pkgs.nix
        ];
        text = ''
          root="''${GARDEN_ROOT:-}"
          if [ -z "$root" ]; then
            root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          fi
          exec nix build "$root#site" "$@"
        '';
      };
      deploy = pkgs.writeShellApplication {
        name = "garden-deploy";
        runtimeInputs = [
          pkgs.bash
          pkgs.coreutils
          pkgs.fd
          pkgs.gitMinimal
          pkgs.git-lfs
          pkgs.openssh
          nodejs
          pnpm
        ];
        text = ''
          deploy_root="''${GARDEN_DEPLOY_ROOT:-$HOME/workspace/aarnphm.github.io}"
          if [ ! -d "$deploy_root/.git" ]; then
            printf '%s\n' "missing deploy checkout: $deploy_root" >&2
            exit 1
          fi
          cd "$deploy_root"
          git pull
          exec bash ./deploy.sh "$@"
        '';
      };
      preview = pkgs.writeShellApplication {
        name = "garden-preview";
        runtimeInputs = [
          pkgs.caddy
          pkgs.gitMinimal
          pkgs.nix
        ];
        text = ''
          root="''${GARDEN_ROOT:-}"
          if [ -z "$root" ]; then
            root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          fi
          site="''${GARDEN_SITE_OUT:-}"
          if [ -z "$site" ]; then
            site="$(nix build "$root#deployable" --no-link --print-out-paths)"
          fi
          exec caddy file-server --listen "''${GARDEN_PREVIEW_ADDR:-localhost:8080}" --root "$site" "$@"
        '';
      };
    in {
      default = app build;
      build = app build;
      build-site = app buildSite;
      deploy = app deploy;
      preview = app preview;
    };
  in {
    packages = forAllSystems mkPackages;
    apps = forAllSystems mkApps;
    checks = forAllSystems (
      system: let
        packages = self.packages.${system};
        preCommitCheck = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            alejandra = {
              enable = true;
              files = "^flake\\.nix$";
            };
            deadnix = {
              enable = true;
              files = "^flake\\.nix$";
            };
            statix = {
              enable = true;
              files = "^flake\\.nix$";
            };
          };
        };
      in {
        inherit (packages) deployable quartz vault;
        pre-commit-check = preCommitCheck;
      }
    );
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
        nodejs = pkgs.nodejs_24;
        pnpm = pkgs.pnpm_10.override {inherit nodejs;};
        preCommitCheck = self.checks.${system}.pre-commit-check;
      in {
        default = pkgs.mkShell {
          packages =
            [
              nodejs
              pnpm
              pkgs.alejandra
              pkgs.cargo
              pkgs.cbfmt
              pkgs.clang
              pkgs.deadnix
              pkgs.dune_3
              pkgs.fd
              pkgs.ghc
              pkgs.git-lfs
              pkgs.go
              pkgs.lua54Packages.lua
              pkgs.nixd
              pkgs.ocaml
              pkgs.ocamlformat_0_28_1
              pkgs.ormolu
              pkgs.ripgrep
              pkgs.rustc
              pkgs.rustfmt
              pkgs.statix
              pkgs.uv
              pkgs.zig
              pkgs.python313
            ]
            ++ preCommitCheck.enabledPackages;
          shellHook = ''
            ${preCommitCheck.shellHook}
            export COREPACK_ENABLE_DOWNLOAD_PROMPT=0
          '';
        };
      }
    );
    formatter = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in
        pkgs.alejandra
    );
  };
}
