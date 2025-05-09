// model-integration.js - اتصال مدل آموزش‌دیده به ربات تلگرام

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const technicalIndicators = require('technicalindicators');
const axios = require('axios');

// مسیر پوشه مدل‌ها
const MODELS_DIR = path.join(__dirname, 'models');
const BINANCE_API = 'https://api.binance.com/api/v3';

// لیست ارزهای دیجیتال
const CRYPTOS = [
  { symbol: 'BTCUSDT', name: 'بیت‌کوین' },
  { symbol: 'ETHUSDT', name: 'اتریوم' },
  { symbol: 'BNBUSDT', name: 'بایننس کوین' },
  { symbol: 'SOLUSDT', name: 'سولانا' },
  { symbol: 'ADAUSDT', name: 'کاردانو' },
  { symbol: 'XRPUSDT', name: 'ریپل' },
  { symbol: 'DOGEUSDT', name: 'دوج‌کوین' },
  { symbol: 'DOTUSDT', name: 'پولکادات' },
  { symbol: 'AVAXUSDT', name: 'اولانچ' },
  { symbol: 'MATICUSDT', name: 'پلیگان' }
];

// کلاس پیش‌بینی‌کننده با مدل آموزش‌دید
class CryptoPredictorAI {
  constructor() {
    this.models = {};
    this.modelConfigs = {};
  }

  // بارگذاری همه مدل‌ها
  async loadAllModels() {
    console.log('در حال بارگذاری مدل‌های آموزش‌دیده...');
    
    for (const crypto of CRYPTOS) {
      try {
        await this.loadModel(crypto.symbol);
        console.log(`✅ مدل ${crypto.symbol} با موفقیت بارگذاری شد.`);
      } catch (error) {
        console.error(`❌ خطا در بارگذاری مدل ${crypto.symbol}: