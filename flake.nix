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
      ./patches
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
      pnpm = pkgs.pnpm_11.override {inherit nodejs;};
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
        hash = "sha256-G8QsfbUGgc0nFDh0iKaDcEa0rl6gEepVuX1fny5K83E=";
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
  in {
    packages = forAllSystems mkPackages;
    checks = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
        # tiktoken reads <cache dir>/<sha1 of the rank URL>, so the rank file is fetched at build
        # time and the hook never needs the network or uv.
        tiktokenCache =
          pkgs.runCommand "tiktoken-cache" {
            rank = pkgs.fetchurl {
              url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken";
              sha256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";
            };
          } ''
            mkdir -p "$out"
            cp "$rank" "$out/fb374d419588a4632f3f557e76b4b70aebbca790"
          '';
        tokenBudget = pkgs.writeShellApplication {
          name = "token-budget";
          runtimeInputs = [
            pkgs.gitMinimal
            (pkgs.python313.withPackages (ps: [ps.tiktoken]))
          ];
          text = ''
            export TIKTOKEN_CACHE_DIR=${tiktokenCache}
            # The flake check runs hooks over a sandbox copy with an empty index and no harness path.
            root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
            script="$root/harness/token-budget/check.py"
            [ -f "$script" ] || exit 0
            exec python3 "$script"
          '';
        };
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
            token-budget = {
              enable = true;
              name = "token-budget";
              entry = "${tokenBudget}/bin/token-budget";
              language = "system";
              pass_filenames = false;
              always_run = true;
              stages = ["pre-commit"];
            };
          };
        };
      in {
        pre-commit-check = preCommitCheck;
      }
    );
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
        nodejs = pkgs.nodejs_24;
        pnpm = pkgs.pnpm_11.override {inherit nodejs;};
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
