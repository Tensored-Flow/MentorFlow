"use client";

import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowLeft, Play, CheckCircle, XCircle, Shuffle } from "lucide-react";
import { useState, useEffect } from "react";
import questions from "@/data/questions.json";

const subjectData: Record<string, { title: string; color: string; tasks: string[] }> = {
  programming: {
    title: "Programming",
    color: "from-purple-500 to-purple-600",
    tasks: ["Variable Tracing", "If Conditions", "List Indexing", "Boolean Logic", "Python Syntax", "Java Syntax", "JavaScript Syntax", "SQL Queries", "Rust Syntax", "HTML/CSS", "Bash Commands", "Loop Tracing"],
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

type QuestionEntry = {
  id: string;
  subject: string;
  topic: string;
  difficulty: string;
  prompt: string;
  choices: string[];
  answer_index: number;
  explanation: string;
};

const pickQuestion = (subject: string, previousId?: string): QuestionEntry => {
  const subjectQuestions = (questions as QuestionEntry[]).filter((entry) => entry.subject === subject);
  if (!subjectQuestions.length) {
    return (questions as QuestionEntry[])[0];
  }
  let candidate = subjectQuestions[Math.floor(Math.random() * subjectQuestions.length)];
  if (previousId && subjectQuestions.length > 1) {
    while (candidate.id === previousId) {
      candidate = subjectQuestions[Math.floor(Math.random() * subjectQuestions.length)];
    }
  }
  return candidate;
};

export default function SubjectPracticePage() {
  const params = useParams();
  const subject = params.subject as string;
  const data = subjectData[subject];

  const [started, setStarted] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [question, setQuestion] = useState(() => pickQuestion(subject));

  useEffect(() => {
    setQuestion(prev => pickQuestion(subject, prev.prompt));
    setStarted(false);
    setSelectedAnswer(null);
    setShowResult(false);
  }, [subject]);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">Subject not found</p>
      </div>
    );
  }

  const handleNext = () => {
    setQuestion((prev) => pickQuestion(subject, prev.prompt));
    setSelectedAnswer(null);
    setShowResult(false);
  };

  const handleAnswer = (index: number) => {
    if (showResult) return;
    setSelectedAnswer(index);
    setShowResult(true);
  };

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
              <p className="text-gray-400">{data.tasks.length} task types available</p>
            </div>
          </div>
        </div>

        {!started ? (
          /* Task Type Selection */
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h2 className="text-xl font-semibold mb-4">Select a Task Type</h2>
            <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3 mb-8">
              {data.tasks.map((task, index) => (
                <motion.button
                  key={task}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.03 }}
                  onClick={() => setStarted(true)}
                  className="p-4 rounded-xl glass hover:bg-white/10 transition-all text-left group"
                >
                  <span className="text-sm font-medium group-hover:text-purple-400 transition-colors">{task}</span>
                </motion.button>
              ))}
            </div>

            <div className="text-center">
              <p className="text-gray-500 mb-4">Or start with a random task</p>
              <button
                onClick={() => setStarted(true)}
                className="inline-flex items-center px-6 py-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 font-semibold hover:opacity-90 transition-opacity"
              >
                <Shuffle className="w-5 h-5 mr-2" />
                Random Task
              </button>
            </div>
          </motion.div>
        ) : (
          /* Question View */
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto"
          >
            <div className="glass rounded-2xl p-8 mb-6">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Question</span>
                <span className="text-xs px-3 py-1 rounded-full bg-purple-500/20 text-purple-400">
                  Difficulty: Medium
                </span>
              </div>

              <p className="text-xl font-medium mb-8 font-mono bg-black/30 p-4 rounded-lg">
                {question.prompt}
              </p>

              <div className="space-y-3">
                {question.choices.map((choice, index) => {
                  let buttonClass = "w-full p-4 rounded-xl text-left transition-all ";
                  if (showResult) {
                    if (index === question.answer_index) {
                      buttonClass += "bg-green-500/20 border-2 border-green-500 text-green-400";
                    } else if (index === selectedAnswer) {
                      buttonClass += "bg-red-500/20 border-2 border-red-500 text-red-400";
                    } else {
                      buttonClass += "glass opacity-50";
                    }
                  } else {
                    buttonClass += "glass hover:bg-white/10 hover:border-purple-500/50 border-2 border-transparent";
                  }

                  return (
                    <button
                      key={index}
                      onClick={() => handleAnswer(index)}
                      disabled={showResult}
                      className={buttonClass}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono">{choice}</span>
                        {showResult && index === question.answer_index && (
                          <CheckCircle className="w-5 h-5 text-green-400" />
                        )}
                        {showResult && index === selectedAnswer && index !== question.answer_index && (
                          <XCircle className="w-5 h-5 text-red-400" />
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex flex-col gap-3 md:flex-row md:justify-center">
              {showResult && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-center space-x-4"
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
