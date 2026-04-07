import { Activity, LogOut, UserCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Navbar({ user, onLogout }) {
  return (
    <motion.nav 
      initial={{ y: -100 }} animate={{ y: 0 }}
      className="fixed top-0 w-full glass-panel border-b-0 border-t-0 border-l-0 border-r-0 border-b-white/10 z-50 px-8 py-4 flex justify-between items-center rounded-none"
    >
      <div className="flex items-center gap-3">
        <Activity className="text-cyan-400 w-6 h-6" />
        <span className="text-xl font-bold tracking-tight text-white">NeuroRIS</span>
      </div>
      
      {user && (
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-3 border-r border-white/10 pr-6">
            <UserCircle className="w-8 h-8 text-slate-400" />
            <div>
              <p className="text-sm font-semibold text-white leading-tight">{user.name}</p>
              <p className="text-xs text-slate-400">{user.email}</p>
            </div>
          </div>
          
          <button 
            onClick={onLogout} 
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-red-400 transition-colors font-medium"
          >
            <LogOut className="w-4 h-4" /> Disconnect
          </button>
        </div>
      )}
    </motion.nav>
  );
}