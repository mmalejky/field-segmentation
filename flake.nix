{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-22.05"; # for python 3.7
  };
  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      fhs = pkgs.buildFHSUserEnv {
        name = "fhs-shell";
        targetPkgs = pkgs: with pkgs; [
          python37Full
          #python39Packages.pip
          #python39Packages.virtualenv
          #python39Packages.tkinter

          git
          gitRepo
          gnupg
          autoconf
          curl
          procps
          gnumake
          util-linux
          m4
          gperf
          unzip
          cudaPackages_11_6.cudatoolkit
          linuxPackages.nvidia_x11
          libGLU libGL
          xorg.libXi xorg.libXmu freeglut
          xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
          ncurses5
          stdenv.cc
		  stdenv.cc.cc.lib
          binutils
          glib
        ];
        multiPkgs = pkgs: with pkgs; [ zlib ];
        profile = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
          source .venv/bin/activate
          pip install torch==1.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/
          pip install matplotlib
          pip install opencv-python
          pip install albumentations
          pip install kornia==0.6.8
          pip install importlib-metadata
          pip install tqdm
          pip install jupyter
          # For MaskRCNN
          pip install torchvision==0.10.0
          pip install pycocotools
        '';
      };
    in { devShells.${system}.default = fhs.env; };
}
