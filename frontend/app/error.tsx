"use client";

import Link from "next/link";
import { useEffect } from "react";

export default function GlobalError({ error, reset }: { error: Error; reset: () => void }) {
  useEffect(() => {
    console.error("Unhandled error in app router:", error);
  }, [error]);

  return (
    <div className="min-h-screen bg-[#020013] text-white flex flex-col items-center justify-center px-6 py-12 text-center">
      <p className="text-sm uppercase tracking-[0.5em] text-white/60 mb-2">MentorFlow</p>
      <h1 className="text-4xl font-bold mb-4">Something went wrong.</h1>
      <p className="text-base text-white/70 max-w-xl mb-6">
        An unexpected error occurred while rendering the page. You can try refreshing or return to the homepage.
      </p>
      <div className="flex gap-4">
        <button
          onClick={() => reset()}
          className="px-6 py-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 font-semibold shadow-lg"
        >
          Refresh
        </button>
        <Link href="/" className="px-6 py-3 rounded-full border border-white/30 text-white font-semibold">
          Go Home
        </Link>
      </div>
    </div>
  );
}
