import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-white/10 mt-20">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="text-xl font-bold gradient-text">
              MentorFlow
            </Link>
            <p className="mt-3 text-gray-400 text-sm">
              Where AI learns to teach — and teaches you too.
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="font-semibold mb-4">Product</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li><Link href="/practice" className="hover:text-white transition-colors">Practice</Link></li>
              <li><Link href="/training" className="hover:text-white transition-colors">Training</Link></li>
              <li><Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link></li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="font-semibold mb-4">Company</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li><Link href="/about" className="hover:text-white transition-colors">About</Link></li>
              <li><a href="https://github.com" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a></li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="font-semibold mb-4">Account</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li><Link href="/login" className="hover:text-white transition-colors">Login</Link></li>
              <li><Link href="/register" className="hover:text-white transition-colors">Register</Link></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-white/10 mt-12 pt-8 text-center text-gray-500 text-sm">
          © {new Date().getFullYear()} MentorFlow. Built with ❤️ and RL.
        </div>
      </div>
    </footer>
  );
}
