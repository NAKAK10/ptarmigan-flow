class PtarmiganFlow < Formula
  desc "Push-to-talk transcription daemon for macOS using Moonshine"
  homepage "https://github.com/NAKAK10/ptarmigan-flow"
  # stable-release: updated by GitHub Actions on release publish.
  # stable-release-start
  url "https://github.com/NAKAK10/ptarmigan-flow/archive/refs/tags/v0.3.11.tar.gz"
  sha256 "5d27d28298e0f3724b7778c020bd15abedb96dbc7521b48dfdb99f102a94e5bb"
  version "0.3.11"
  # stable-release-end
  head "https://github.com/NAKAK10/ptarmigan-flow.git", branch: "main"
  preserve_rpath
  skip_clean "libexec/README.md"
  skip_clean "libexec/pyproject.toml"
  skip_clean "libexec/uv.lock"

  depends_on "portaudio"
  depends_on "python@3.11"
  depends_on "uv"

  def install
    libexec.install buildpath.children

    python, uv = resolve_python_and_uv!
    ENV["UV_PYTHON"] = python.to_s
    ENV["UV_PYTHON_DOWNLOADS"] = "never"
    ENV["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PTARMIGAN_FLOW"] = version.to_s unless build.head?
    system uv, "sync", "--project", libexec, "--frozen"

    %w[ptarmigan-flow pflow].each do |command_name|
      (bin/command_name).write <<~SH
        #!/bin/bash
        exec "#{python}" "#{opt_libexec}/src/ptarmigan_flow/homebrew_bootstrap.py" \
          --libexec "#{opt_libexec}" \
          --var-dir "#{var}/ptarmigan-flow" \
          --python "#{python}" \
          --uv "#{uv}" \
          -- \
          "$@"
      SH
      chmod 0755, bin/command_name
    end
  end

  private

  def resolve_python_and_uv!
    primary_python = Formula["python@3.11"].opt_bin/"python3.11"
    primary_uv = Formula["uv"].opt_bin/"uv"
    fallback_python = Pathname.new("/opt/homebrew/opt/python@3.11/bin/python3.11")
    fallback_uv = Pathname.new("/opt/homebrew/opt/uv/bin/uv")
    if prefer_fallback_toolchain?(primary_python, fallback_python, fallback_uv)
      opoo <<~EOS
        Detected x86_64 Homebrew python@3.11 at #{primary_python}; using arm64 fallback #{fallback_python}.
      EOS
      return [fallback_python, fallback_uv]
    end

    return [primary_python, primary_uv] if python_is_healthy?(primary_python)

    if fallback_python.exist? && fallback_uv.exist? && python_is_healthy?(fallback_python)
      opoo <<~EOS
        Detected broken python@3.11 at #{primary_python}; using fallback #{fallback_python}.
      EOS
      return [fallback_python, fallback_uv]
    end

    odie <<~EOS
      Unable to locate a healthy Python 3.11 runtime for ptarmigan-flow install.
      Checked:
        - #{primary_python}
        - #{fallback_python}

      Try:
        brew reinstall python@3.11
        ./scripts/install_brew.sh
    EOS
  end

  def prefer_fallback_toolchain?(primary_python, fallback_python, fallback_uv)
    return false unless primary_python.exist?
    return false unless fallback_python.exist? && fallback_uv.exist?
    return false unless python_is_healthy?(primary_python)
    return false unless python_is_healthy?(fallback_python)

    python_arch(primary_python) == "x86_64" && python_arch(fallback_python) == "arm64"
  end

  def python_is_healthy?(python)
    return false unless python.exist?

    probe = Utils.safe_popen_read(
      python.to_s,
      "-c",
      "import platform; print(platform.mac_ver()[0])",
    ).strip
    !probe.empty?
  rescue ErrorDuringExecution, Errno::ENOENT, RuntimeError
    false
  end

  def python_arch(python)
    return "" unless python.exist?

    Utils.safe_popen_read(
      python.to_s,
      "-c",
      "import platform; print(platform.machine())",
    ).strip
  rescue ErrorDuringExecution, Errno::ENOENT, RuntimeError
    ""
  end

  test do
    assert_match "ptarmigan-flow", shell_output("#{bin}/ptarmigan-flow --help")
    assert_match "ptarmigan-flow", shell_output("#{bin}/pflow --help")
    assert_predicate opt_prefix/"README.md", :exist?
    probe = shell_output(
      <<~EOS
        #{opt_libexec}/.venv/bin/python -c "import ctypes; import moonshine_voice; from pathlib import Path; lib = Path(moonshine_voice.__file__).resolve().with_name('libmoonshine.dylib'); ctypes.CDLL(str(lib)); print('moonshine-runtime-ok')"
      EOS
    )
    assert_match "moonshine-runtime-ok", probe
  end
end
