class MoonshineFlow < Formula
  desc "Push-to-talk transcription daemon for macOS using Moonshine"
  homepage "https://github.com/NAKAK10/moonshine-flow"
  # stable-release: updated by GitHub Actions on release publish.
  # stable-release-start
  url "https://github.com/NAKAK10/moonshine-flow/archive/refs/tags/v0.1.3.tar.gz"
  sha256 "6f3013f629e0d556bd625510f3070ce5f7b2e60b4b9d3d8e46264dfccb087c72"
  version "0.1.3"
  # stable-release-end
  head "https://github.com/NAKAK10/moonshine-flow.git", branch: "main"
  preserve_rpath
  skip_clean "libexec/README.md"
  skip_clean "libexec/pyproject.toml"
  skip_clean "libexec/uv.lock"

  depends_on "portaudio"
  depends_on "python@3.11"
  depends_on "uv"

  def install
    libexec.install buildpath.children

    python = Formula["python@3.11"].opt_bin/"python3.11"
    uv = Formula["uv"].opt_bin/"uv"
    ENV["UV_PYTHON"] = python
    ENV["UV_PYTHON_DOWNLOADS"] = "never"
    ENV["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MOONSHINE_FLOW"] = version.to_s unless build.head?
    system uv, "sync", "--project", libexec, "--frozen"

    %w[moonshine-flow mflow].each do |command_name|
      (bin/command_name).write <<~SH
        #!/bin/bash
        exec "#{python}" "#{opt_libexec}/src/moonshine_flow/homebrew_bootstrap.py" \
          --libexec "#{opt_libexec}" \
          --var-dir "#{var}/moonshine-flow" \
          --python "#{python}" \
          --uv "#{uv}" \
          -- \
          "$@"
      SH
      chmod 0755, bin/command_name
    end
  end

  test do
    assert_match "moonshine-flow", shell_output("#{bin}/moonshine-flow --help")
    assert_match "moonshine-flow", shell_output("#{bin}/mflow --help")
    assert_predicate opt_prefix/"README.md", :exist?
    probe = shell_output(
      <<~EOS
        #{opt_libexec}/.venv/bin/python -c "import ctypes; import moonshine_voice; from pathlib import Path; lib = Path(moonshine_voice.__file__).resolve().with_name('libmoonshine.dylib'); ctypes.CDLL(str(lib)); print('moonshine-runtime-ok')"
      EOS
    )
    assert_match "moonshine-runtime-ok", probe
  end
end
