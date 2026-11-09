import type { Metrik } from "../api";
import { fmt } from "../format";

const ITEMS: { key: keyof Metrik; label: string }[] = [
  { key: "p5", label: "P@5" },
  { key: "p10", label: "P@10" },
  { key: "ndcg10", label: "nDCG@10" },
];

export default function MetricsTable({ final }: { final: Metrik }) {
  return (
    <section aria-label="Metrik untuk query ini" className="grid grid-cols-3 gap-3">
      {ITEMS.map(({ key, label }) => (
        <div key={key} className="rounded-xl bg-surface p-4 shadow-sm">
          <p className="text-sm text-ink-2">{label}</p>
          <p className="text-2xl font-semibold">{fmt(final[key])}</p>
        </div>
      ))}
    </section>
  );
}
