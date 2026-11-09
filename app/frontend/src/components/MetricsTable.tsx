import type { Metrik } from "../api";
import { fmt, fmtSigned } from "../format";

const ROWS: { key: keyof Metrik; label: string }[] = [
  { key: "p5", label: "P@5" },
  { key: "p10", label: "P@10" },
  { key: "ndcg10", label: "nDCG@10" },
];

export default function MetricsTable({ bm25, final }: { bm25: Metrik; final: Metrik }) {
  return (
    <section className="overflow-x-auto rounded-xl bg-surface p-4 shadow-sm">
      <h2 className="mb-2 text-sm font-semibold">Metrik untuk query ini</h2>
      <table className="tnum w-full min-w-[420px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-ink-2">
            <th className="py-1.5 font-medium">Metrik</th>
            <th className="py-1.5 text-right font-medium">BM25</th>
            <th className="py-1.5 text-right font-medium">Final stacking</th>
            <th className="py-1.5 text-right font-medium">Selisih</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(({ key, label }) => {
            const d = final[key] - bm25[key];
            return (
              <tr key={key} className="border-b border-line/60 last:border-0">
                <td className="py-1.5">{label}</td>
                <td className="py-1.5 text-right">{fmt(bm25[key])}</td>
                <td className="py-1.5 text-right font-semibold">{fmt(final[key])}</td>
                <td className={`py-1.5 text-right ${d > 0 ? "text-green-800" : d < 0 ? "text-red-700" : "text-ink-2"}`}>
                  {fmtSigned(d)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
