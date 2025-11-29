"use client";

import { motion } from "framer-motion";

const testimonials = [
  {
    quote: "MentorFlow helped me practise the same tasks the AI learns — crazy cool.",
    author: "Alex Chen",
    role: "CS Student",
    avatar: "AC",
  },
  {
    quote: "The adaptive curriculum feels smarter than any study app I've used.",
    author: "Sarah Kim",
    role: "Data Scientist",
    avatar: "SK",
  },
  {
    quote: "Watching the teacher RL agent optimise the curriculum is wild.",
    author: "James Wright",
    role: "ML Engineer",
    avatar: "JW",
  },
  {
    quote: "Finally, a learning platform that actually adapts to how I learn.",
    author: "Maria Garcia",
    role: "PhD Researcher",
    avatar: "MG",
  },
];

export default function Testimonials() {
  return (
    <section className="py-24 px-6 bg-gradient-to-b from-transparent via-purple-900/10 to-transparent">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Loved by <span className="gradient-text">Learners</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            See what others are saying about their MentorFlow experience.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.author}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="p-6 rounded-2xl glass"
            >
              <p className="text-gray-300 mb-6 italic">"{testimonial.quote}"</p>
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-sm font-semibold">
                  {testimonial.avatar}
                </div>
                <div>
                  <p className="font-semibold text-sm">{testimonial.author}</p>
                  <p className="text-gray-500 text-sm">{testimonial.role}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
