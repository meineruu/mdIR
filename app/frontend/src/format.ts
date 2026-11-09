export const fmt = (n: number, digits = 3) =>
  n.toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fmtSigned = (n: number, digits = 3) => `${n > 0 ? "+" : ""}${fmt(n, digits)}`;

export const fmtP = (p: number) => (p < 0.001 ? p.toExponential(1).replace(".", ",") : fmt(p, 3));
