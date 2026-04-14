import { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, ArrowRight, Lock, Mail, AlertCircle, UserCircle, ClipboardCopy } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';

// Replace with your Codespace URL in production
const API_URL = 'http://127.0.0.1:8000'; 
const TEST_EMAIL = 'test@ris.local';
const TEST_PASSWORD = 'Test@12345';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await axios.post(`${API_URL}/api/login`, { email, password });
      onLogin(res.data.token, res.data.user);
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
    }
    setLoading(false);
  };

  const handleGuestAccess = () => {
    onLogin('guest', { name: 'Guest User', email: 'Free dashboard access' });
    navigate('/dashboard');
  };

  const fillTestCredentials = () => {
    setEmail(TEST_EMAIL);
    setPassword(TEST_PASSWORD);
  };

  return (
    <div className="min-h-screen flex items-center justify-center pt-16 px-4">
      <motion.div 
        initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md p-10 glass-panel rounded-[2rem]"
      >
        <div className="flex flex-col items-center mb-8">
          <motion.div 
            initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2, type: "spring" }}
            className="p-4 bg-cyan-500/10 rounded-2xl mb-6 border border-cyan-500/30 shadow-[0_0_30px_rgba(6,182,212,0.2)]"
          >
            <Activity className="w-8 h-8 text-cyan-400" />
          </motion.div>
          <h1 className="text-3xl font-bold text-white tracking-tight">NeuroRIS Portal</h1>
          <p className="text-slate-400 mt-2 text-sm">Secure radiologist authentication</p>
        </div>

        {error && (
          <div className="mb-6 p-3 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3 text-red-400 text-sm">
            <AlertCircle className="w-5 h-5" /> {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-5">
          <div className="relative group">
            <Mail className="absolute left-4 top-3.5 w-5 h-5 text-slate-500 group-focus-within:text-cyan-400 transition-colors" />
            <input type="email" required value={email} onChange={e => setEmail(e.target.value)} className="w-full bg-black/40 border border-white/10 focus:border-cyan-500 rounded-xl py-3 pl-12 pr-4 text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all" placeholder="Email Address" />
          </div>

          <div className="relative group">
            <Lock className="absolute left-4 top-3.5 w-5 h-5 text-slate-500 group-focus-within:text-cyan-400 transition-colors" />
            <input type="password" required value={password} onChange={e => setPassword(e.target.value)} className="w-full bg-black/40 border border-white/10 focus:border-cyan-500 rounded-xl py-3 pl-12 pr-4 text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all" placeholder="Password" />
          </div>

          <motion.button 
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold py-3.5 rounded-xl transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] mt-8 disabled:opacity-50"
          >
            {loading ? 'Authenticating...' : 'Access Dashboard'} <ArrowRight className="w-5 h-5" />
          </motion.button>
        </form>

        <div className="mt-4 space-y-3">
          <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4 text-sm text-slate-300">
            <div className="flex items-center gap-2 text-cyan-300 font-semibold mb-2">
              <ClipboardCopy className="w-4 h-4" /> Test credentials
            </div>
            <p>Email: <span className="font-mono text-white">{TEST_EMAIL}</span></p>
            <p>Password: <span className="font-mono text-white">{TEST_PASSWORD}</span></p>
            <button
              type="button"
              onClick={fillTestCredentials}
              className="mt-3 inline-flex items-center gap-2 text-xs font-semibold text-cyan-300 hover:text-cyan-200"
            >
              <UserCircle className="w-4 h-4" /> Fill test login fields
            </button>
          </div>
          <button
            type="button"
            onClick={handleGuestAccess}
            className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 text-white font-semibold py-3.5 rounded-xl transition-all border border-white/10"
          >
            <UserCircle className="w-5 h-5" /> Continue as Guest
          </button>
          <p className="text-center text-slate-500 text-xs">Guest access opens the dashboard without a login.</p>
        </div>

        <p className="text-center text-slate-400 mt-8 text-sm">
          No account? <Link to="/signup" className="text-cyan-400 hover:text-cyan-300 font-medium">Request access</Link>
        </p>
      </motion.div>
    </div>
  );
}