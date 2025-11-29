"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Code, Calculator, Atom, FlaskConical } from "lucide-react";

const subjects = [
  {
    id: "programming",
    title: "Programming",
    description: "Python, Java, JavaScript, SQL, and more",
    icon: Code,
    color: "from-purple-500 to-purple-600",
    tasks: 12,
  },
  {
    id: "maths",
    title: "Mathematics",
    description: "Algebra, calculus, and problem solving",
    icon: Calculator,
    color: "from-blue-500 to-blue-600",
    tasks: 8,
  },
  {
    id: "physics",
    title: "Physics",
    description: "Kinematics, forces, and energy",
    icon: Atom,
    color: "from-green-500 to-green-600",
    tasks: 6,
  },
  {
    id: "chemistry",
    title: "Chemistry",
    description: "Moles, balancing equations, and reactions",
    icon: FlaskConical,
    color: "from-orange-500 to-orange-600",
    tasks: 6,
  },
];

export default function SubjectGrid() {
  return (
    <div className="grid md:grid-cols-2 gap-6">
      {subjects.map((subject, index) => (
        <motion.div
          key={subject.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <Link
            href={`/practice/${subject.id}`}
            className="block p-6 rounded-2xl glass hover:bg-white/10 transition-all duration-300 group"
          >
            <div className="flex items-start space-x-4">
              <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${subject.color} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                <subject.icon className="w-7 h-7 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold mb-1">{subject.title}</h3>
                <p className="text-gray-400 text-sm mb-3">{subject.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">{subject.tasks} task types</span>
                  <span className="text-purple-400 text-sm font-medium group-hover:translate-x-1 transition-transform">
                    Start Practising →
                  </span>
                </div>
              </div>
            </div>
          </Link>
        </motion.div>
      ))}
    </div>
  );
}
