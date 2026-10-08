const $ = (id) => document.getElementById(id);
const form = $("form");

// Indian-style price formatting: Cr / Lakh
function formatINR(n) {
  if (n >= 1e7) return "₹" + (n / 1e7).toFixed(2) + " Cr";
  if (n >= 1e5) return "₹" + (n / 1e5).toFixed(2) + " Lakh";
  return "₹" + Math.round(n).toLocaleString("en-IN");
}

function fillSelect(id, items, selected) {
  $(id).innerHTML = items
    .map((v) => `<option value="${v}" ${v === selected ? "selected" : ""}>${v}</option>`)
    .join("");
}

function showError(msg) {
  const el = $("error");
  el.textContent = msg;
  el.hidden = !msg;
}

async function init() {
  try {
    const opts = await (await fetch("/api/options")).json();
    fillSelect("city", opts.cities, "Pune");
    fillSelect("property_type", opts.property_types, "Apartment");
    fillSelect("parking", opts.parking, "Covered");
    fillSelect("furnishing", opts.furnishing, "Semi-furnished");
    fillSelect("facing", opts.facing, "East");
    const m = await (await fetch("/api/metrics")).json();
    $("model").textContent =
      `Model: ${m.best_model} · R² ${m.r2} · typical error about ${m.mape_pct}%`;
  } catch (e) {
    showError("Could not load the form options. Refresh the page and try again.");
  }
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showError("");
  const data = Object.fromEntries(new FormData(form).entries());
  data.has_lift = form.has_lift.checked ? 1 : 0;
  data.has_security = form.has_security.checked ? 1 : 0;

  const btn = $("submit");
  btn.disabled = true;
  btn.textContent = "Estimating…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || "Something went wrong.");

    $("price").textContent = formatINR(out.price);
    $("psf").textContent = `About ₹${out.price_per_sqft.toLocaleString("en-IN")} per sq ft`;
    $("low").textContent = formatINR(out.low);
    $("high").textContent = formatINR(out.high);
    const pos = ((out.price - out.low) / (out.high - out.low)) * 100;
    $("marker").style.left = Math.min(Math.max(pos, 5), 95) + "%";
    $("empty").hidden = true;
    $("output").hidden = false;
    if (window.innerWidth <= 860) {
      $("output").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Estimate price";
  }
});

init();
