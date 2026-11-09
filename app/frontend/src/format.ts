export const fmt = (n: number, digits = 3) =>
  n.toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fmtSigned = (n: number, digits = 3) => `${n > 0 ? "+" : ""}${fmt(n, digits)}`;

export const fmtP = (p: number) => (p < 0.001 ? p.toExponential(1).replace(".", ",") : fmt(p, 3));

/** Warna tetap per sinyal: identitas mengikuti sinyal, bukan peringkatnya. */
export const SIGNAL_COLORS: Record<string, string> = {
  exp: "#2a78d6",
  knn: "#eb6834",
  rrf: "#1baf7a",
  co: "#eda100",
  rm3: "#e87ba4",
  ce_base: "#008300",
};
