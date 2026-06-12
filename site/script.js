/* Copy-to-clipboard for command blocks. No dependencies, no tracking. */
(function () {
  "use strict";

  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-200vw";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(area);
    return ok;
  }

  function showDone(button) {
    var label = button.textContent;
    button.textContent = button.getAttribute("data-done") || "Copied";
    button.classList.add("copied");
    window.setTimeout(function () {
      button.textContent = label;
      button.classList.remove("copied");
    }, 1600);
  }

  document.querySelectorAll(".cmd .copy-btn").forEach(function (button) {
    button.addEventListener("click", function () {
      var code = button.parentElement.querySelector("code");
      if (!code) {
        return;
      }
      var text = code.innerText.replace(/\n+$/, "") + "\n";
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () {
            showDone(button);
          },
          function () {
            if (fallbackCopy(text)) {
              showDone(button);
            }
          }
        );
      } else if (fallbackCopy(text)) {
        showDone(button);
      }
    });
  });
})();
