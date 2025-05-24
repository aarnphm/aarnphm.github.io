{
  description = "Quartz and adjacent";

  inputs = {
    # system
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-schemas.url = "https://flakehub.com/f/DeterminateSystems/flake-schemas/*.tar.gz";

    # github
    quartz-src = {
      url = "github:jackyzha0/quartz/v4";
      flake = false;
    };

    # utilities
    git-hooks = {
      url = "github:cachix/git-hooks.nix/master";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  # meta function: https://github.com/NixOS/nixpkgs/blob/master/lib/meta.nix
  outputs = {
    self,
    nixpkgs,
    git-hooks,
    quartz-src,
    flake-schemas,
    ...
  }: let
    packageJson = builtins.fromJSON (builtins.readFile ./package.json);
    version = packageJson.version;

    overlays = {
      default = final: prev: {
        quartz = final.buildNpmPackage {
          pname = "quartz";
          src = quartz-src;
          version = (builtins.fromJSON (builtins.readFile (quartz-src + "/package.json"))).version;
          dontNpmBuild = true;
        };

        quartz-local = final.stdenv.mkDerivation (finalAttrs: {
          name = "quartz-local";
          pname = "quartz";
          inherit version;
          src = self;

          nativeBuildInputs = with final; [
            nodejs
            pnpm.configHook
            npmHooks.npmInstallHook
          ];

          pnpmDeps = final.pnpm.fetchDeps {
            inherit (finalAttrs) pname version src;
            hash = "sha256-FI45pzI0kELSE3992MxWjRcs/UBqTmZ8RkCe2n41nfY=";
          };

          installPhase = ''
            runHook preInstall
            nodejsInstallExecutables "${./package.json}"
            runHook postInstall
          '';

          meta = {
            mainProgram = finalAttrs.pname;
            description = "flake of ${finalAttrs.pname} v${finalAttrs.version}";
          };
        });
      };
    };

    forPackages = system:
      import nixpkgs {
        inherit system;
        overlays = [overlays.default];
        config = {
          allowUnfree = true;
          allowBroken = !(builtins.elem system nixpkgs.lib.platforms.darwin);
          allowUnsupportedSystem = true;
        };
      };
  in
    builtins.foldl' nixpkgs.lib.recursiveUpdate {} (
      builtins.map (
        system: let
          pkgs = forPackages system;
          mkApp = {
            drv,
            name ? drv.pname or drv.name,
            exePath ? drv.passthru.exePath or "/bin/${name}",
          }: {
            type = "app";
            program = "${drv}${exePath}";
          };
        in {
          apps.${system} = builtins.listToAttrs (
            builtins.map (
              name: {
                inherit name;
                value =
                  mkApp {drv = pkgs.${name};}
                  // {
                    meta = {
                      mainProgram = name;
                      description = "flake of ${name} v${version}";
                    };
                  };
              }
            ) [
              "quartz"
              "quartz-local"
            ]
          );

          packages.${system} = {
            default = pkgs.quartz;
            quartz = pkgs.quartz;
            quartz-local = pkgs.quartz-local;
          };

          checks.${system} = {
            pre-commit-check = git-hooks.lib.${system}.run {
              src = builtins.path {
                path = ./.;
                name = "source";
              };
              hooks = {
                alejandra.enable = true;
                statix.enable = true;
              };
            };
          };

          devShells.${system} = {
            default = pkgs.mkShell {
              inherit (self.checks.${system}.pre-commit-check) shellHook;
              buildInputs = self.checks.${system}.pre-commit-check.enabledPackages;
              packages = with pkgs; [
                nodejs_24
                prettierd
                eslint_d
              ];
            };
          };
        }
      ) ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"]
    )
    // {
      inherit overlays;
      schemas = flake-schemas.schemas;
    };
}
