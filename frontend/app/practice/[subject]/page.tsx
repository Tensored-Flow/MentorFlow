"use client";

import { useParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { ArrowLeft, Play, CheckCircle, XCircle, Shuffle, Loader2, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import SkillDifficultyBar from "@/components/SkillDifficultyBar";
import { fetchNextTask, submitAttempt, TaskPayload } from "@/lib/taskApi";

const subjectData: Record<string, { title: string; color: string; tasks: string[] }> = {
  programming: {
    title: "Programming",
    color: "from-purple-500 to-purple-600",
    tasks: ["Variable Tracing", "If Conditions", "List Indexing", "Boolean Logic", "Python/Java/JS/C++/Rust Syntax", "SQL Queries", "HTML/CSS", "Bash Commands", "Loop Tracing", "Regex"],
  },
  maths: {
    title: "Mathematics",
    color: "from-blue-500 to-blue-600",
    tasks: ["Arithmetic", "Algebra", "Fractions", "Percentages", "Equations", "Exponents", "Logarithms", "Probability"],
  },
  physics: {
    title: "Physics",
    color: "from-green-500 to-green-600",
    tasks: ["Kinematics", "Newton's Laws", "Energy", "Momentum", "Waves", "Electricity"],
  },
  chemistry: {
    title: "Chemistry",
    color: "from-orange-500 to-orange-600",
    tasks: ["Moles", "Balancing Equations", "Atomic Structure", "Periodic Table", "Acids & Bases", "Stoichiometry"],
  },
};

const difficultyLabels = ["Beginner", "Intermediate", "Advanced", "Expert", "Olympiad++"];

export default function SubjectPracticePage() {
  const params = useParams();
  const subject = params.subject as string;
  const data = subjectData[subject];

  const [started, setStarted] = useState(false);
  const [question, setQuestion] = useState<TaskPayload | null>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(false);
  const [difficultyLevel, setDifficultyLevel] = useState(1);
  const [rollingAccuracy, setRollingAccuracy] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [justLeveledUp, setJustLeveledUp] = useState(false);

  const loadQuestion = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchNextTask(subject);
      setQuestion(payload);
      setDifficultyLevel(payload.difficulty_level ?? 1);
      setRollingAccuracy(payload.rolling_accuracy ?? 0);
      setSelectedAnswer(null);
      setShowResult(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to fetch question");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setStarted(false);
    setQuestion(null);
    setSelectedAnswer(null);
    setShowResult(false);
    setDifficultyLevel(1);
    setRollingAccuracy(0);
    setError(null);
  }, [subject]);

  const handleStart = async () => {
    setStarted(true);
    await loadQuestion();
  };

  const handleAnswer = async (index: number) => {
    if (!question || showResult) return;
    setSelectedAnswer(index);
    setShowResult(true);
    const correct = index === question.answer_index;
    try {
      const response = await submitAttempt({ correct });
      const nextLevel = response.difficulty_state.current_level ?? difficultyLevel;
      setJustLeveledUp(nextLevel > difficultyLevel);
      setDifficultyLevel(nextLevel);
      setRollingAccuracy(response.difficulty_state.rolling_accuracy ?? rollingAccuracy);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to record attempt");
    }
  };

  const handleNext = async () => {
    await loadQuestion();
    setJustLeveledUp(false);
  };

  const difficultyLabel = useMemo(() => difficultyLabels[difficultyLevel - 1] ?? "Adaptive", [difficultyLevel]);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">Subject not found</p>
      </div>
    );
  }

  return (
    <section className="min-h-screen pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-6">
        {/* Header */}
        <div className="mb-8">
          <Link href="/practice" className="inline-flex items-center text-gray-400 hover:text-white mb-4 transition-colors">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to subjects
          </Link>
          <div className="flex items-center space-x-4">
            <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${data.color} flex items-center justify-center`}>
              <Play className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">{data.title}</h1>
              <p className="text-gray-400">{data.tasks.length} task types available · Adaptive difficulty</p>
            </div>
          </div>
        </div>

        {!started ? (
          /* Task Type Selection */
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <h2 className="text-xl font-semibold mb-4">Select a Task Type</h2>
            <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3 mb-8">
              {data.tasks.map((task, index) => (
                <motion.button
                  key={task}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.03 }}
                  onClick={handleStart}
                  className="p-4 rounded-xl glass hover:bg-white/10 transition-all text-left group"
                >
                  <span className="text-sm font-medium group-hover:text-purple-400 transition-colors">{task}</span>
                </motion.button>
              ))}
            </div>

            <div className="text-center">
              <p className="text-gray-500 mb-4">Or start with a random task</p>
              <button
                onClick={handleStart}
                className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 font-semibold hover:opacity-90 transition-opacity"
              >
                <Shuffle className="w-5 h-5 mr-2" />
                Random Task
              </button>
            </div>
          </motion.div>
        ) : (
          /* Question View */
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto space-y-4">
            <SkillDifficultyBar level={difficultyLevel} rollingAccuracy={rollingAccuracy} />

            <AnimatePresence>
              {justLeveledUp && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-green-500/10 text-green-300 border border-green-500/30"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Level up! Now at Level {difficultyLevel} ({difficultyLabel})</span>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="glass rounded-2xl p-8 mb-3 border border-white/10">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 uppercase tracking-[0.3em]">Question</span>
                <span className="text-xs px-3 py-1 rounded-full bg-purple-500/20 text-purple-200 border border-purple-500/40">
                  Difficulty Level: {difficultyLevel}/5 · {difficultyLabel}
                </span>
              </div>
              {difficultyLevel >= 4 && (
                <p className="text-xs text-amber-300 mt-2">Trick mode: expect deceptive edge cases.</p>
              )}

              {loading && (
                <div className="flex items-center justify-center py-10 text-white/70">
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Loading next adaptive task...
                </div>
              )}

              {!loading && question && (
                <>
                  <p className="text-xl font-medium mb-6 font-mono bg-black/30 p-4 rounded-lg">
                    {question.prompt}
                  </p>
                  <p className="text-sm text-white/60 mb-4">Topic: {question.topic ?? question.family ?? "Adaptive Task"}</p>

                  <div className="space-y-3">
                    {question.choices.map((choice, index) => {
                      const isCorrect = index === question.answer_index;
                      const isChosen = index === selectedAnswer;
                      let buttonClass = "w-full p-4 rounded-xl text-left transition-all ";
                      if (showResult) {
                        if (isCorrect) {
                          buttonClass += "bg-green-500/20 border-2 border-green-500 text-green-300";
                        } else if (isChosen) {
                          buttonClass += "bg-red-500/20 border-2 border-red-500 text-red-300";
                        } else {
                          buttonClass += "glass opacity-70";
                        }
                      } else {
                        buttonClass += "glass hover:bg-white/10 hover:border-purple-500/50 border-2 border-transparent";
                      }

                      return (
                        <button key={index} onClick={() => handleAnswer(index)} disabled={showResult || loading} className={buttonClass}>
                          <div className="flex items-center justify-between">
                            <span className="font-mono">{choice}</span>
                            {showResult && isCorrect && <CheckCircle className="w-5 h-5 text-green-400" />}
                            {showResult && isChosen && !isCorrect && <XCircle className="w-5 h-5 text-red-400" />}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
            </div>

            {error && <p className="text-sm text-red-400">{error}</p>}

            <div className="flex flex-col gap-3 md:flex-row md:justify-center">
              {showResult && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-wrap justify-center gap-3"
                >
                  <button
                    onClick={handleNext}
                    className="px-6 py-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 font-semibold hover:opacity-90 transition-opacity"
                  >
                    Next Question →
                  </button>
                  <button
                    onClick={() => setStarted(false)}
                    className="px-6 py-3 rounded-full glass hover:bg-white/10 transition-colors"
                  >
                    Change Task Type
                  </button>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </section>
  );
}
