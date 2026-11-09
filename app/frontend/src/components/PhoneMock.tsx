/** Pratinjau statis dari hasil nyata query PLAIN-2670 (P@5 BM25 0,200; Final 1,000). */
const RESULTS = [
  "Carcinogenicity and regulation of caramel colorings.",
  "Toxicity and carcinogenicity studies of 4-methylimidazole in F344/N rats …",
  "Does 4-methylimidazole have tumor preventive activity in the rat?",
];

export default function PhoneMock() {
  return (
    <div className="relative hidden h-[520px] overflow-hidden lg:block" aria-hidden>
      <div className="absolute inset-x-6 top-4 mx-auto h-[640px] max-w-[320px] rotate-2 rounded-[3rem] border-[10px] border-[#23262b] bg-white shadow-2xl">
        <div className="flex items-center justify-between px-6 pt-3 text-xs font-semibold">
          <span>9:41</span>
          <span className="h-6 w-24 rounded-full bg-[#14161a]" />
          <span className="text-brand">5G</span>
        </div>
        <div className="px-5 pt-6">
          <p className="text-xs font-medium text-brand">Contoh hasil nyata · split test</p>
          <h3 className="mt-1 font-serif text-2xl font-semibold leading-tight">Is Caramel Color Carcinogenic?</h3>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <div className="rounded-2xl bg-page p-3">
              <p className="text-[11px] text-ink-2">P@5 BM25</p>
              <p className="text-xl font-semibold">0,200</p>
            </div>
            <div className="rounded-2xl bg-brand p-3 text-white">
              <p className="text-[11px] opacity-85">P@5 Final</p>
              <p className="text-xl font-semibold">1,000</p>
            </div>
          </div>
          <ol className="mt-4 space-y-3">
            {RESULTS.map((title, i) => (
              <li key={title} className="flex gap-2 text-sm leading-snug">
                <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-blue-100 text-[11px] font-semibold text-brand">{i + 1}</span>
                <span>
                  {title}
                  <span className="mt-0.5 block text-xs font-medium text-green-800">✓ Relevan di qrels</span>
                </span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}
