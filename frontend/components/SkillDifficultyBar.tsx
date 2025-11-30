import { motion } from "framer-motion";
import { Info } from "lucide-react";

const labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Olympiad++"];
const gradient = "linear-gradient(90deg, #10b981 0%, #f6c344 25%, #f97316 50%, #ef4444 75%, #a855f7 100%)";

type Props = {
  level: number;
  rollingAccuracy?: number;
  compact?: boolean;
};

const clampLevel = (level: number) => Math.min(5, Math.max(1, Math.round(level)));

export default function SkillDifficultyBar({ level, rollingAccuracy, compact }: Props) {
  const safeLevel = clampLevel(level);
  const percent = (safeLevel / 5) * 100;
  const label = labels[safeLevel - 1];

  return (
    <div className={`w-full ${compact ? "" : "glass"} rounded-2xl px-4 py-3 border border-white/10`}>
      <div className="flex items-center justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <p className="text-xs uppercase tracking-[0.25em] text-white/60">Difficulty</p>
          <div className="text-xs text-white/50 flex items-center gap-1">
            <Info className="w-4 h-4" />
            <span>Difficulty increases as you improve accuracy.</span>
          </div>
        </div>
        <div className="text-sm font-semibold text-white">
          Level {safeLevel}/5 · {label}
        </div>
      </div>

      <div className="h-3 w-full rounded-full bg-white/10 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ type: "spring", stiffness: 140, damping: 18 }}
          className="h-full rounded-full"
          style={{ backgroundImage: gradient }}
        />
      </div>

      <div className="flex justify-between text-[11px] text-white/60 mt-2">
        {labels.map((lbl, idx) => (
          <span key={lbl} className={idx + 1 === safeLevel ? "text-white" : ""}>
            {idx + 1}. {lbl}
          </span>
        ))}
      </div>

      {rollingAccuracy !== undefined && (
        <p className="text-[11px] text-white/60 mt-1">Rolling accuracy (last 20): {(rollingAccuracy * 100).toFixed(1)}%</p>
      )}
    </div>
  );
}
