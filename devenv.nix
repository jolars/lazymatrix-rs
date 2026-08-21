{
  pkgs,
  ...
}:
{
  env.OPENBLAS_LP64_LIB = "${pkgs.openblasCompat}/lib";

  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    liteparse
    cargo-llvm-cov
    cargo-flamegraph
    cargo-audit
    cargo-deny
    cargo-msrv
    gnuplot
    samply
    pprof
    wasm-pack
    perf
    go-task
    quartoMinimal
    shfmt
  ];

  languages = {
    rust = {
      enable = true;
      toolchainFile = ./rust-toolchain.toml;
    };
  };

  git-hooks = {
    hooks = {
      clippy = {
        enable = true;

        settings = {
          allFeatures = true;
        };
      };

      rustfmt = {
        enable = true;
      };
    };
  };
}
