"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X } from "lucide-react";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const links = [
    { href: "/", label: "Home" },
    { href: "/practice", label: "Practice" },
    { href: "/training", label: "Training" },
    { href: "/login", label: "Login" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <span className="text-2xl font-bold gradient-text">MentorFlow</span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center space-x-8">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-gray-300 hover:text-white transition-colors duration-200"
              >
                {link.label}
              </Link>
            ))}
            <Link
              href="/register"
              className="px-5 py-2 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-white font-medium hover:opacity-90 transition-opacity"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden text-white"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Nav */}
        {isOpen && (
          <div className="md:hidden mt-4 pb-4 space-y-4">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="block text-gray-300 hover:text-white transition-colors"
                onClick={() => setIsOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            <Link
              href="/register"
              className="block w-full text-center px-5 py-2 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-white font-medium"
              onClick={() => setIsOpen(false)}
            >
              Get Started
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}
