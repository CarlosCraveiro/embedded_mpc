{
  description = "Dev shell with g++, onnxruntime, qpOASES e Eigen (nixpkgs 25.05)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            gcc          # fornece g++
            onnxruntime
            qpoases
            eigen
          ];
        };
      }
    );
}
