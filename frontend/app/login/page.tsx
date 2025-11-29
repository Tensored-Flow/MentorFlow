"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Mail, Lock, ArrowRight } from "lucide-react";

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6 pt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        {/* Card */}
        <div className="p-8 rounded-2xl glass relative overflow-hidden">
          {/* Gradient border effect */}
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500 to-blue-500 opacity-20" />
          
          <div className="relative z-10">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold mb-2">Welcome Back</h1>
              <p className="text-gray-400">Sign in to continue learning</p>
            </div>

            <form className="space-y-5">
              {/* Email */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="email"
                    placeholder="you@example.com"
                    className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-purple-500 focus:outline-none transition-colors"
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="password"
                    placeholder="••••••••"
                    className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-purple-500 focus:outline-none transition-colors"
                  />
                </div>
              </div>

              {/* Forgot password */}
              <div className="text-right">
                <Link href="/forgot-password" className="text-sm text-purple-400 hover:text-purple-300">
                  Forgot password?
                </Link>
              </div>

              {/* Submit */}
              <button
                type="submit"
                className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold hover:opacity-90 transition-opacity flex items-center justify-center space-x-2"
              >
                <span>Sign In</span>
                <ArrowRight className="w-5 h-5" />
              </button>
            </form>

            {/* Divider */}
            <div className="flex items-center my-6">
              <div className="flex-1 h-px bg-white/10" />
              <span className="px-4 text-sm text-gray-500">or</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            {/* Google */}
            <button className="w-full py-3 rounded-xl border border-white/10 hover:bg-white/5 transition-colors flex items-center justify-center space-x-2">
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              <span>Continue with Google</span>
            </button>

            {/* Register link */}
            <p className="text-center mt-6 text-gray-400">
              Don&apos;t have an account?{" "}
              <Link href="/register" className="text-purple-400 hover:text-purple-300">
                Register
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
