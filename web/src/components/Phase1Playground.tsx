"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Cpu,
  FileText,
  Zap,
  Send,
  Loader2,
  Copy,
  Check,
  AlertCircle,
  Sliders,
  Brain,
  BarChart3,
} from "lucide-react";

type ModelType = "classifier" | "lm" | "summarizer";

interface ClassifyResult {
  category: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface GenerateResult {
  generated_text: string;
  tokens_generated: number;
}

interface SummarizeResult {
  summary: string;
  original_length: number;
  summary_length: number;
}

export default function Phase1Playground() {
  const [activeModel, setActiveModel] = useState<ModelType>("classifier");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Model-specific state
  const [classifyResult, setClassifyResult] = useState<ClassifyResult | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateResult | null>(null);
  const [summarizeResult, setSummarizeResult] = useState<SummarizeResult | null>(null);

  // Generation parameters
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);

  const models = [
    {
      id: "classifier" as const,
      name: "Text Classifier",
      icon: Cpu,
      description: "Encoder-only transformer for paper categorization",
      color: "blue",
      placeholder: "Enter a research paper abstract to classify into categories (CV, ML, NLP, Accel, WS)...",
    },
    {
      id: "lm" as const,
      name: "Language Model",
      icon: FileText,
      description: "Decoder-only transformer for text generation",
      color: "green",
      placeholder: "Enter a prompt to generate research paper text...",
    },
    {
      id: "summarizer" as const,
      name: "Summarizer",
      icon: Zap,
      description: "Encoder-decoder transformer for TL;DR generation",
      color: "purple",
      placeholder: "Enter a research paper abstract to summarize...",
    },
  ];

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setError(null);

    try {
      let endpoint = "";
      let body: Record<string, unknown> = {};

      switch (activeModel) {
        case "classifier":
          endpoint = "/api/phase1/classify";
          body = { text: input };
          break;
        case "lm":
          endpoint = "/api/phase1/generate";
          body = { prompt: input, max_length: maxLength, temperature, top_k: topK };
          break;
        case "summarizer":
          endpoint = "/api/phase1/summarize";
          body = { text: input, max_length: maxLength };
          break;
      }

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "API request failed");
      }

      const data = await res.json();

      switch (activeModel) {
        case "classifier":
          setClassifyResult(data);
          break;
        case "lm":
          setGenerateResult(data);
          break;
        case "summarizer":
          setSummarizeResult(data);
          break;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getColorClasses = (color: string) => ({
    bg: `bg-${color}-500/20`,
    border: `border-${color}-500/30`,
    text: `text-${color}-400`,
    gradient: `from-${color}-600/20 to-${color}-800/20`,
  });

  const activeModelData = models.find((m) => m.id === activeModel)!;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">
          Phase 1: Transformer Playground
        </h1>
        <p className="text-gray-400">
          Interact with custom transformer models built from scratch
        </p>
      </div>

      {/* Model Selector */}
      <div className="flex flex-wrap gap-3 justify-center">
        {models.map((model) => (
          <motion.button
            key={model.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              setActiveModel(model.id);
              setClassifyResult(null);
              setGenerateResult(null);
              setSummarizeResult(null);
              setError(null);
            }}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl border transition-all ${
              activeModel === model.id
                ? `bg-${model.color}-500/20 border-${model.color}-500/50 text-white`
                : "bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600"
            }`}
          >
            <model.icon className="w-5 h-5" />
            <span className="font-medium">{model.name}</span>
          </motion.button>
        ))}
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <activeModelData.icon className={`w-5 h-5 text-${activeModelData.color}-400`} />
              <h2 className="font-semibold text-white">{activeModelData.name}</h2>
            </div>
            <p className="text-gray-400 text-sm mb-4">{activeModelData.description}</p>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={activeModelData.placeholder}
              className="w-full h-48 bg-gray-900/50 border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
            />

            {/* Generation Parameters (for LM and Summarizer) */}
            {(activeModel === "lm" || activeModel === "summarizer") && (
              <div className="mt-4 p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                <div className="flex items-center gap-2 mb-4">
                  <Sliders className="w-4 h-4 text-gray-400" />
                  <span className="text-sm font-medium text-gray-300">Parameters</span>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      Max Length: {maxLength}
                    </label>
                    <input
                      type="range"
                      min="20"
                      max="300"
                      value={maxLength}
                      onChange={(e) => setMaxLength(Number(e.target.value))}
                      className="w-full accent-blue-500"
                    />
                  </div>
                  
                  {activeModel === "lm" && (
                    <>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">
                          Temperature: {temperature.toFixed(1)}
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="2.0"
                          step="0.1"
                          value={temperature}
                          onChange={(e) => setTemperature(Number(e.target.value))}
                          className="w-full accent-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">
                          Top-K: {topK}
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="100"
                          value={topK}
                          onChange={(e) => setTopK(Number(e.target.value))}
                          className="w-full accent-blue-500"
                        />
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !input.trim()}
              className="mt-4 w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-xl hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  {activeModel === "classifier" && "Classify"}
                  {activeModel === "lm" && "Generate"}
                  {activeModel === "summarizer" && "Summarize"}
                </>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}

          {/* Results Display */}
          {classifyResult && activeModel === "classifier" && (
            <ClassificationResult result={classifyResult} />
          )}

          {generateResult && activeModel === "lm" && (
            <GenerationResult 
              result={generateResult} 
              onCopy={() => copyToClipboard(generateResult.generated_text)}
              copied={copied}
            />
          )}

          {summarizeResult && activeModel === "summarizer" && (
            <SummarizationResult 
              result={summarizeResult}
              onCopy={() => copyToClipboard(summarizeResult.summary)}
              copied={copied}
            />
          )}
        </div>

        {/* Info Panel */}
        <div className="space-y-4">
          <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-blue-400" />
              Model Info
            </h3>
            
            <div className="space-y-3 text-sm">
              <InfoRow label="Architecture" value={
                activeModel === "classifier" ? "Encoder-only" :
                activeModel === "lm" ? "Decoder-only" : "Encoder-Decoder"
              } />
              <InfoRow label="d_model" value="256" />
              <InfoRow label="Layers" value={activeModel === "summarizer" ? "4 + 4" : "4"} />
              <InfoRow label="Heads" value="4" />
              <InfoRow label="Dropout" value={
                activeModel === "classifier" ? "0.3" :
                activeModel === "lm" ? "0.15" : "0.2"
              } />
              <InfoRow label="Vocab Size" value="8,000" />
              <InfoRow label="Max Seq Len" value="256" />
            </div>
          </div>

          <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-green-400" />
              Performance
            </h3>
            
            <div className="space-y-3 text-sm">
              {activeModel === "classifier" && (
                <>
                  <InfoRow label="Val Accuracy" value="78.1%" />
                  <InfoRow label="Categories" value="5 classes" />
                  <InfoRow label="Training" value="Early stopped @ 11" />
                </>
              )}
              {activeModel === "lm" && (
                <>
                  <InfoRow label="Perplexity" value="46.12" />
                  <InfoRow label="Training" value="30 epochs" />
                  <InfoRow label="Data" value="10,500 abstracts" />
                </>
              )}
              {activeModel === "summarizer" && (
                <>
                  <InfoRow label="ROUGE-1" value="0.3137" />
                  <InfoRow label="ROUGE-2" value="0.1526" />
                  <InfoRow label="ROUGE-L" value="0.2777" />
                </>
              )}
            </div>
          </div>

          {/* Sample Prompts */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6">
            <h3 className="font-semibold text-white mb-4">Sample Inputs</h3>
            <div className="space-y-2">
              {getSampleInputs(activeModel).map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(sample)}
                  className="w-full text-left p-3 bg-gray-900/50 hover:bg-gray-900 rounded-lg text-gray-400 text-sm transition-colors border border-transparent hover:border-gray-700"
                >
                  {sample.slice(0, 80)}...
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-400">{label}</span>
      <span className="text-white font-medium">{value}</span>
    </div>
  );
}

function ClassificationResult({ result }: { result: ClassifyResult }) {
  const categoryColors: Record<string, string> = {
    CV: "bg-blue-500",
    ML: "bg-green-500",
    NLP: "bg-purple-500",
    Accel: "bg-orange-500",
    WS: "bg-pink-500",
  };

  const categoryNames: Record<string, string> = {
    CV: "Computer Vision",
    ML: "Machine Learning",
    NLP: "Natural Language Processing",
    Accel: "Hardware Acceleration",
    WS: "Web Services",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6"
    >
      <h3 className="font-semibold text-white mb-4">Classification Result</h3>
      
      {/* Main Result */}
      <div className="flex items-center gap-4 mb-6">
        <div className={`px-4 py-2 rounded-xl ${categoryColors[result.category]} text-white font-bold text-lg`}>
          {result.category}
        </div>
        <div>
          <div className="text-white font-medium">{categoryNames[result.category]}</div>
          <div className="text-gray-400 text-sm">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Probability Distribution */}
      <div className="space-y-2">
        <div className="text-sm text-gray-400 mb-2">Probability Distribution</div>
        {Object.entries(result.probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([cat, prob]) => (
            <div key={cat} className="flex items-center gap-3">
              <span className="w-12 text-gray-400 text-sm">{cat}</span>
              <div className="flex-1 h-3 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${prob * 100}%` }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  className={`h-full ${categoryColors[cat]}`}
                />
              </div>
              <span className="w-16 text-right text-gray-400 text-sm">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
      </div>
    </motion.div>
  );
}

function GenerationResult({ 
  result, 
  onCopy, 
  copied 
}: { 
  result: GenerateResult;
  onCopy: () => void;
  copied: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-white">Generated Text</h3>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-400">
            {result.tokens_generated} tokens generated
          </span>
          <button
            onClick={onCopy}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4 text-gray-400" />
            )}
          </button>
        </div>
      </div>
      
      <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
        <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">
          {result.generated_text}
        </p>
      </div>
    </motion.div>
  );
}

function SummarizationResult({ 
  result, 
  onCopy, 
  copied 
}: { 
  result: SummarizeResult;
  onCopy: () => void;
  copied: boolean;
}) {
  const compressionRatio = ((1 - result.summary_length / result.original_length) * 100).toFixed(0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-white">TL;DR Summary</h3>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-400">
            {compressionRatio}% compression
          </span>
          <button
            onClick={onCopy}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4 text-gray-400" />
            )}
          </button>
        </div>
      </div>
      
      <div className="p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl border border-purple-500/20">
        <p className="text-gray-300 leading-relaxed">
          {result.summary}
        </p>
      </div>

      <div className="mt-4 flex gap-4 text-sm text-gray-400">
        <span>Original: {result.original_length} chars</span>
        <span>Summary: {result.summary_length} chars</span>
      </div>
    </motion.div>
  );
}

function getSampleInputs(model: ModelType): string[] {
  switch (model) {
    case "classifier":
      return [
        "We propose a novel attention mechanism for image classification that leverages multi-scale feature representations. Our method achieves state-of-the-art results on ImageNet and COCO benchmarks.",
        "This paper introduces a new approach to natural language understanding using transformer architectures with improved tokenization strategies for multilingual text processing.",
        "We present an efficient hardware accelerator design for deep neural network inference on edge devices, achieving 10x speedup with minimal power consumption.",
      ];
    case "lm":
      return [
        "In this paper, we propose",
        "Large language models have shown remarkable",
        "Recent advances in transformer architectures",
      ];
    case "summarizer":
      return [
        "Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. However, their massive parameter counts often lead to significant computational and memory requirements, making deployment challenging in resource-constrained environments. In this paper, we propose a novel compression technique that reduces model size by 75% while maintaining 95% of the original performance.",
        "Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models with external knowledge. By combining the generative capabilities of LLMs with relevant document retrieval, RAG systems can produce more accurate and factual responses. We present an improved RAG architecture that addresses key limitations in existing approaches.",
      ];
    default:
      return [];
  }
}
