"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, Home, Cpu, MessageSquare, Github, ExternalLink } from "lucide-react";

interface HealthStatus {
  status: string;
  phase1_loaded: boolean;
  phase2_loaded: boolean;
  device: string;
}

interface HeaderProps {
  activeTab: "home" | "phase1" | "phase2";
  setActiveTab: (tab: "home" | "phase1" | "phase2") => void;
  health: HealthStatus | null;
}

export default function Header({ activeTab, setActiveTab, health }: HeaderProps) {
  const tabs = [
    { id: "home" as const, label: "Home", icon: Home },
    { id: "phase1" as const, label: "Phase 1: Transformers", icon: Cpu },
    { id: "phase2" as const, label: "Phase 2: RAG", icon: MessageSquare },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-lg bg-gray-900/80 border-b border-gray-800">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-white">Paper Companion AI</h1>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span className={`w-2 h-2 rounded-full ${health?.status === 'healthy' ? 'bg-green-400' : 'bg-gray-500'}`} />
                {health?.device || 'Offline'}
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                  activeTab === tab.id
                    ? "text-white"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg border border-blue-500/30"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </button>
            ))}
          </nav>

          {/* Mobile Navigation */}
          <nav className="flex md:hidden items-center gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`p-2 rounded-lg transition-colors ${
                  activeTab === tab.id
                    ? "text-white bg-blue-500/20"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <tab.icon className="w-5 h-5" />
              </button>
            ))}
          </nav>

          {/* GitHub Link */}
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white transition-colors"
          >
            <Github className="w-5 h-5" />
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </div>
    </header>
  );
}
