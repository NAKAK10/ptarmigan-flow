/* Copy-to-clipboard for command blocks. No dependencies, no tracking. */
(function () {
  "use strict";

  var prefersReducedMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- copy-to-clipboard ---------- */

  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-200vw";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
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
      if (!code) return;
      var text = code.innerText.replace(/\n+$/, "") + "\n";
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () { showDone(button); },
          function () { if (fallbackCopy(text)) showDone(button); }
        );
      } else if (fallbackCopy(text)) {
        showDone(button);
      }
    });
  });

  /* ---------- scroll fade-in ----------
     Hero elements animate via CSS keyframes (hero-up).
     Everything else uses IntersectionObserver for staggered entry.
  ---------------------------------------- */

  if (prefersReducedMotion || !("IntersectionObserver" in window)) return;

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  /* Exclude .cta-card (lives in .hero and has CSS entrance animation) */
  document.querySelectorAll(
    ".section, .feature, .steps li, .install-card, .perm-card"
  ).forEach(function (el) {
    el.classList.add("anim-target");
    observer.observe(el);
  });
})();

/* ---------- Hero waveform canvas ----------
   Oscilloscope-style sine waves + spectrum analyser bars.
   Runs on every .hero-canvas element found on the page.
----------------------------------------------------- */
(function () {
  "use strict";

  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  if (!window.requestAnimationFrame) return;

  document.querySelectorAll(".hero-canvas").forEach(function (canvas) {
    var ctx = canvas.getContext("2d");
    if (!ctx) return;

    var lw = 0, lh = 0, tick = 0, raf = null;

    function setup() {
      var dpr = window.devicePixelRatio || 1;
      lw = canvas.offsetWidth;
      lh = canvas.offsetHeight;
      if (!lw || !lh) return;
      canvas.width  = Math.round(lw * dpr);
      canvas.height = Math.round(lh * dpr);
    }

    function draw() {
      raf = requestAnimationFrame(draw);
      if (!lw || !lh) return;

      var dpr   = window.devicePixelRatio || 1;
      var dark  = window.matchMedia("(prefers-color-scheme: dark)").matches;
      /* Alpine glass palette: mid-tone steel blues with alpha read on both
         light and dark backgrounds; one rust "comb" line at low opacity. */
      var STEEL  = [90, 130, 160];   /* ice-steel blue        */
      var ALPINE = [68, 112, 143];   /* steel blue (--alpine) */
      var FROST  = [127, 168, 201];  /* lighter ice           */
      var COMB   = [199, 92, 58];    /* rust comb accent      */
      var aScale = dark ? 1.0 : 0.75;
      var W = lw, H = lh;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      /* center line of the oscilloscope */
      var cy = H * 0.48;

      /* ── Waveform layers ──────────────────────────── */
      /* [freq, amplitude, speed, alpha, lineWidth, glowBlur, phaseOffset, rgb] */
      var waves = [
        [1.1,  H * 0.19, 0.006, 0.07, 2.5,  0,  0.0, STEEL],
        [2.3,  H * 0.12, 0.011, 0.24, 2.2, 16,  0.5, ALPINE],
        [3.7,  H * 0.07, 0.018, 0.14, 1.5,  0,  1.1, FROST],
        [5.4,  H * 0.04, 0.027, 0.10, 1.0,  0,  2.3, COMB],
        [0.7,  H * 0.10, 0.004, 0.06, 2.0,  0,  3.7, STEEL],
      ];

      for (var wi = 0; wi < waves.length; wi++) {
        var wv = waves[wi];
        var freq = wv[0], amp = wv[1], speed = wv[2],
            alpha = wv[3] * aScale, lwidth = wv[4],
            glow  = wv[5], ph = wv[6], rgb = wv[7];
        var col = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ",";

        var grd = ctx.createLinearGradient(0, 0, W, 0);
        grd.addColorStop(0,    col + "0)");
        grd.addColorStop(0.05, col + alpha + ")");
        grd.addColorStop(0.95, col + alpha + ")");
        grd.addColorStop(1,    col + "0)");

        ctx.save();
        ctx.strokeStyle = grd;
        ctx.lineWidth   = lwidth;
        if (glow) {
          ctx.shadowBlur  = glow;
          ctx.shadowColor = col + "0.6)";
        }

        ctx.beginPath();
        for (var x = 0; x <= W; x += 2) {
          var y = cy + amp * Math.sin((x / W) * Math.PI * 2 * freq + tick * speed * 60 + ph);
          if (x === 0) ctx.moveTo(x, y);
          else         ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }

      /* ── Spectrum analyser bars ───────────────────── */
      var N      = Math.max(28, Math.floor(W / 10));
      var slot   = W / N;
      var barW   = slot * 0.52;
      var maxBH  = H * 0.24;
      var baseY  = H * 0.975;

      for (var i = 0; i < N; i++) {
        var bx = i * slot + slot * 0.5;

        /* bass-heavy left side bias — simulates real spectrum */
        var lowBoost = 0.55 + 0.45 * Math.pow(1 - i / N, 1.2);

        var h1 = Math.abs(Math.sin(i * 0.43  + tick * 0.041));
        var h2 = Math.abs(Math.sin(i * 0.81  + tick * 0.065 + 1.2));
        var h3 = Math.abs(Math.sin(i * 0.26  + tick * 0.029 + 2.4));
        var h4 = Math.abs(Math.sin(i * 1.37  + tick * 0.085 + 0.6));
        var bh = maxBH * lowBoost * (h1 * 0.38 + h2 * 0.30 + h3 * 0.20 + h4 * 0.12);
        bh = Math.max(bh, maxBH * 0.04);

        /* smooth edge fade-out */
        var ef = Math.min(i / (N * 0.07), 1) * Math.min((N - 1 - i) / (N * 0.07), 1);
        var bAlpha = (dark ? 0.30 : 0.20) * ef;
        var bc = STEEL;

        var bGrd = ctx.createLinearGradient(0, baseY - bh, 0, baseY);
        bGrd.addColorStop(0, "rgba(" + bc[0] + "," + bc[1] + "," + bc[2] + "," + bAlpha + ")");
        bGrd.addColorStop(1, "rgba(" + bc[0] + "," + bc[1] + "," + bc[2] + ",0)");
        ctx.fillStyle = bGrd;
        ctx.fillRect(bx - barW / 2, baseY - bh, barW, bh);

        /* peak dot on each bar */
        if (dark && ef > 0.5) {
          ctx.fillStyle = "rgba(" + bc[0] + "," + bc[1] + "," + bc[2] + "," + (bAlpha * 1.5) + ")";
          ctx.fillRect(bx - barW / 2, baseY - bh - 2, barW, 2);
        }
      }

      tick++;
    }

    /* Debounced resize */
    var resizeTimer;
    function onResize() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { setup(); }, 80);
    }

    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(onResize).observe(canvas.parentElement || canvas);
    }
    window.addEventListener("resize", onResize);

    /* Pause when tab is hidden */
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        cancelAnimationFrame(raf);
        raf = null;
      } else if (!raf) {
        draw();
      }
    });

    setup();
    draw();
  });
})();
