import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import Signup from './components/Signup';

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('ris_token'));
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('ris_user')));

  const loginUser = (tokenData, userData) => {
    localStorage.setItem('ris_token', tokenData);
    localStorage.setItem('ris_user', JSON.stringify(userData));
    setToken(tokenData);
    setUser(userData);
  };

  const logoutUser = () => {
    localStorage.removeItem('ris_token');
    localStorage.removeItem('ris_user');
    setToken(null);
    setUser(null);
  };

  return (
    <Router>
      <div className="mesh-bg" />
      <div className="min-h-screen relative z-10">
        <Navbar user={user} onLogout={logoutUser} />
        <Routes>
          <Route path="/" element={!token ? <Login onLogin={loginUser} /> : <Navigate to="/dashboard" />} />
          <Route path="/signup" element={!token ? <Signup onLogin={loginUser} /> : <Navigate to="/dashboard" />} />
          <Route path="/dashboard" element={token ? <Dashboard token={token} /> : <Navigate to="/" />} />
        </Routes>
      </div>
    </Router>
  );
}