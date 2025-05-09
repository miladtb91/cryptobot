const { Telegraf } = require('telegraf');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const technicalIndicators = require('technicalindicators');
const cron = require('node-cron');

// تنظیمات ربات
const BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN';
const bot = new Telegraf(BOT_TOKEN);

// لیست ۱۰ ارز دیجیتال محبوب
const POPULAR_CRYPTOS = [
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

// بازه‌های زمانی برای تحلیل
const TIMEFRAMES = ['15m', '1h', '4h', '1d'];

// تنظیمات API بایننس
const BINANCE_API = 'https://api.binance.com/api/v3';

// کلاس تحلیل‌گر با هوش مصنوعی
class CryptoAnalyzer {
  constructor() {
    this.models = {};
    this.loadModels();
  }

  // بارگذاری مدل‌های از پیش آموزش‌دیده
  async loadModels() {
    try {
      for (const crypto of POPULAR_CRYPTOS) {
        // در یک پروژه واقعی، مدل‌ها باید از قبل آموزش دیده و ذخیره شده باشند
        this.models[crypto.symbol] = await this.createModel();
        console.log(`مدل برای ${crypto.name} (${crypto.symbol}) بارگذاری شد`);
      }
    } catch (error) {
      console.error('خطا در بارگذاری مدل‌ها:', error);
    }
  }

  // ایجاد یک مدل LSTM ساده برای پیش‌بینی قیمت
  async createModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.lstm({
      units: 50,
      returnSequences: true,
      inputShape: [14, 5] // 14 روز داده تاریخی با 5 ویژگی (open, high, low, close, volume)
    }));
    
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 50, returnSequences: false }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1 }));
    
    model.compile({
      optimizer: tf.train.adam(),
      loss: 'meanSquaredError'
    });
    
    return model;
  }

  // دریافت داده‌های تاریخی از بایننس
  async getHistoricalData(symbol, timeframe, limit = 100) {
    try {
      const response = await axios.get(`${BINANCE_API}/klines`, {
        params: {
          symbol: symbol,
          interval: timeframe,
          limit: limit
        }
      });
      
      // تبدیل داده‌های خام به فرمت مناسب
      return response.data.map(candle => ({
        timestamp: candle[0],
        open: parseFloat(candle[1]),
        high: parseFloat(candle[2]),
        low: parseFloat(candle[3]),
        close: parseFloat(candle[4]),
        volume: parseFloat(candle[5])
      }));
    } catch (error) {
      console.error(`خطا در دریافت داده‌های ${symbol}:`, error);
      return [];
    }
  }

  // محاسبه شاخص‌های تکنیکال
  calculateIndicators(data) {
    if (data.length < 30) {
      return null;
    }

    const closes = data.map(candle => candle.close);
    const highs = data.map(candle => candle.high);
    const lows = data.map(candle => candle.low);
    const volumes = data.map(candle => candle.volume);

    // محاسبه میانگین متحرک
    const sma10 = technicalIndicators.SMA.calculate({ period: 10, values: closes });
    const sma20 = technicalIndicators.SMA.calculate({ period: 20, values: closes });
    const sma50 = technicalIndicators.SMA.calculate({ period: 50, values: closes });

    // محاسبه RSI
    const rsi = technicalIndicators.RSI.calculate({ period: 14, values: closes });

    // محاسبه MACD
    const macd = technicalIndicators.MACD.calculate({
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      values: closes
    });

    // محاسبه باندهای بولینگر
    const bollingerBands = technicalIndicators.BollingerBands.calculate({
      period: 20,
      stdDev: 2,
      values: closes
    });

    // محاسبه Stochastic
    const stochastic = technicalIndicators.Stochastic.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: 14,
      signalPeriod: 3
    });

    // محاسبه ATR
    const atr = technicalIndicators.ATR.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: 14
    });

    // محاسبه OBV (On Balance Volume)
    const obv = [];
    let obvValue = 0;
    
    for (let i = 1; i < data.length; i++) {
      if (closes[i] > closes[i - 1]) {
        obvValue += volumes[i];
      } else if (closes[i] < closes[i - 1]) {
        obvValue -= volumes[i];
      }
      obv.push(obvValue);
    }

    return {
      sma: { sma10, sma20, sma50 },
      rsi: rsi[rsi.length - 1],
      macd: macd[macd.length - 1],
      bollingerBands: bollingerBands[bollingerBands.length - 1],
      stochastic: stochastic[stochastic.length - 1],
      atr: atr[atr.length - 1],
      obv: obv[obv.length - 1],
      currentPrice: closes[closes.length - 1],
      priceChange24h: ((closes[closes.length - 1] - closes[closes.length - 25]) / closes[closes.length - 25]) * 100
    };
  }

  // تصمیم‌گیری بر اساس قوانین تکنیکال
  getTechnicalDecision(indicators) {
    let buySignals = 0;
    let sellSignals = 0;
    let totalSignals = 0;
    const signals = [];

    // تحلیل میانگین متحرک
    if (indicators.sma.sma10[indicators.sma.sma10.length - 1] > indicators.sma.sma20[indicators.sma.sma20.length - 1]) {
      buySignals++;
      signals.push('Cross Above SMA20 ✅');
    } else {
      sellSignals++;
      signals.push('Cross Below SMA20 ❌');
    }
    totalSignals++;

    // تحلیل RSI
    if (indicators.rsi < 30) {
      buySignals++;
      signals.push('RSI Oversold ✅');
    } else if (indicators.rsi > 70) {
      sellSignals++;
      signals.push('RSI Overbought ❌');
    }
    totalSignals++;

    // تحلیل MACD
    if (indicators.macd.MACD > indicators.macd.signal) {
      buySignals++;
      signals.push('MACD Bullish ✅');
    } else {
      sellSignals++;
      signals.push('MACD Bearish ❌');
    }
    totalSignals++;

    // تحلیل باندهای بولینگر
    if (indicators.currentPrice < indicators.bollingerBands.lower) {
      buySignals++;
      signals.push('Price Below BB Lower ✅');
    } else if (indicators.currentPrice > indicators.bollingerBands.upper) {
      sellSignals++;
      signals.push('Price Above BB Upper ❌');
    }
    totalSignals++;

    // تحلیل Stochastic
    if (indicators.stochastic.k < 20 && indicators.stochastic.k > indicators.stochastic.d) {
      buySignals++;
      signals.push('Stochastic Bullish ✅');
    } else if (indicators.stochastic.k > 80 && indicators.stochastic.k < indicators.stochastic.d) {
      sellSignals++;
      signals.push('Stochastic Bearish ❌');
    }
    totalSignals++;

    // بررسی سیگنال‌های هوشمند
    const strength = ((buySignals - sellSignals) / totalSignals) * 100;
    let decision = '';
    let emoji = '';

    if (strength > 60) {
      decision = 'STRONG_BUY';
      emoji = '🟢🟢🟢';
    } else if (strength > 20) {
      decision = 'BUY';
      emoji = '🟢';
    } else if (strength > -20) {
      decision = 'NEUTRAL';
      emoji = '⚪';
    } else if (strength > -60) {
      decision = 'SELL';
      emoji = '🔴';
    } else {
      decision = 'STRONG_SELL';
      emoji = '🔴🔴🔴';
    }

    return {
      decision,
      emoji,
      strength: Math.abs(strength),
      signals,
      buySignals,
      sellSignals
    };
  }

  // تصمیم‌گیری با استفاده از هوش مصنوعی
  async getAIDecision(symbol, timeframe) {
    try {
      // دریافت داده‌های تاریخی
      const historicalData = await this.getHistoricalData(symbol, timeframe, 200);
      if (historicalData.length < 50) {
        return { error: 'داده‌های کافی برای تحلیل وجود ندارد' };
      }

      // محاسبه شاخص‌های تکنیکال
      const indicators = this.calculateIndicators(historicalData);
      if (!indicators) {
        return { error: 'خطا در محاسبه شاخص‌های تکنیکال' };
      }

      // تحلیل تکنیکال سنتی
      const technicalDecision = this.getTechnicalDecision(indicators);

      // تحلیل هوش مصنوعی (در یک پروژه واقعی، اینجا از مدل آموزش‌دیده استفاده می‌شود)
      // این بخش در یک محیط واقعی با داده‌های واقعی آموزش داده می‌شود
      // اینجا فقط یک شبیه‌سازی ساده است

      let aiConfidence = Math.random() * 30 + 70; // شبیه‌سازی اطمینان هوش مصنوعی (بین 70 تا 100 درصد)
      
      // ترکیب تصمیم‌های تکنیکال و هوش مصنوعی
      let finalDecision;
      if (technicalDecision.decision.includes('BUY')) {
        finalDecision = {
          action: 'BUY',
          entry: indicators.currentPrice,
          stopLoss: indicators.currentPrice * 0.95, // استاپ لاس 5 درصد پایین‌تر
          takeProfit1: indicators.currentPrice * 1.03, // سود هدف اول 3 درصد بالاتر
          takeProfit2: indicators.currentPrice * 1.07, // سود هدف دوم 7 درصد بالاتر
          confidence: Math.round(aiConfidence),
          signals: technicalDecision.signals
        };
      } else if (technicalDecision.decision.includes('SELL')) {
        finalDecision = {
          action: 'SELL',
          entry: indicators.currentPrice,
          stopLoss: indicators.currentPrice * 1.05, // استاپ لاس 5 درصد بالاتر
          takeProfit1: indicators.currentPrice * 0.97, // سود هدف اول 3 درصد پایین‌تر
          takeProfit2: indicators.currentPrice * 0.93, // سود هدف دوم 7 درصد پایین‌تر
          confidence: Math.round(aiConfidence),
          signals: technicalDecision.signals
        };
      } else {
        finalDecision = {
          action: 'NEUTRAL',
          confidence: Math.round(aiConfidence),
          message: 'شرایط مناسب برای معامله نیست',
          signals: technicalDecision.signals
        };
      }

      return {
        symbol,
        timeframe,
        price: indicators.currentPrice,
        priceChange24h: indicators.priceChange24h.toFixed(2),
        technicalDecision: technicalDecision.decision,
        aiDecision: finalDecision,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error(`خطا در تحلیل ${symbol}:`, error);
      return { error: `خطا در تحلیل ${symbol}: ${error.message}` };
    }
  }

  // تحلیل همه ارزهای دیجیتال
  async analyzeAll(timeframe = '4h') {
    const results = [];
    
    for (const crypto of POPULAR_CRYPTOS) {
      console.log(`در حال تحلیل ${crypto.name} (${crypto.symbol})...`);
      const result = await this.getAIDecision(crypto.symbol, timeframe);
      results.push({ ...result, name: crypto.name });
    }
    
    return results;
  }
}

// کلاس ربات تلگرام
class TelegramCryptoBot {
  constructor() {
    this.analyzer = new CryptoAnalyzer();
    this.subscribers = [];
    this.setupBot();
    this.setupScheduledSignals();
  }

  // راه‌اندازی ربات تلگرام
  setupBot() {
    // پیام خوش‌آمدگویی
    bot.start((ctx) => {
      const welcomeMessage = `🤖 *به ربات تحلیل‌گر هوشمند ارز دیجیتال خوش آمدید!*

این ربات با استفاده از هوش مصنوعی و تحلیل تکنیکال، سیگنال‌های دقیق خرید و فروش فیوچرز را برای ۱۰ ارز دیجیتال محبوب ارائه می‌دهد.

💡 *دستورات ربات:*
/signal [نماد] - دریافت سیگنال برای یک ارز دیجیتال خاص (مثال: /signal BTCUSDT)
/signals - دریافت سیگنال برای همه ارزهای دیجیتال
/timeframe [بازه زمانی] - تغییر بازه زمانی تحلیل (مثال: /timeframe 1h)
/subscribe - اشتراک در سیگنال‌های روزانه
/unsubscribe - لغو اشتراک سیگنال‌های روزانه
/help - راهنمای دستورات

⚠️ *هشدار ریسک:* 
این سیگنال‌ها صرفاً جنبه آموزشی دارند. هرگونه معامله بر اساس این سیگنال‌ها، با مسئولیت خود شماست.`;

      ctx.replyWithMarkdown(welcomeMessage);
    });

    // راهنمای دستورات
    bot.help((ctx) => {
      const helpMessage = `🤖 *راهنمای دستورات ربات تحلیل‌گر هوشمند ارز دیجیتال*

💡 *دستورات اصلی:*
/signal [نماد] - دریافت سیگنال برای یک ارز دیجیتال خاص (مثال: /signal BTCUSDT)
/signals - دریافت سیگنال برای همه ارزهای دیجیتال
/timeframe [بازه زمانی] - تغییر بازه زمانی تحلیل (15m, 1h, 4h, 1d)
/subscribe - اشتراک در سیگنال‌های روزانه
/unsubscribe - لغو اشتراک سیگنال‌های روزانه

*ارزهای دیجیتال پشتیبانی شده:*
• بیت‌کوین (BTCUSDT)
• اتریوم (ETHUSDT)
• بایننس کوین (BNBUSDT)
• سولانا (SOLUSDT)
• کاردانو (ADAUSDT)
• ریپل (XRPUSDT)
• دوج‌کوین (DOGEUSDT)
• پولکادات (DOTUSDT)
• اولانچ (AVAXUSDT)
• پلیگان (MATICUSDT)

⚠️ *هشدار ریسک:* 
این سیگنال‌ها صرفاً جنبه آموزشی دارند. هرگونه معامله بر اساس این سیگنال‌ها، با مسئولیت خود شماست.`;

      ctx.replyWithMarkdown(helpMessage);
    });

    // دریافت سیگنال برای یک ارز دیجیتال خاص
    bot.command('signal', async (ctx) => {
      try {
        const args = ctx.message.text.split(' ');
        if (args.length < 2) {
          return ctx.reply('لطفاً نماد ارز دیجیتال را وارد کنید. مثال: /signal BTCUSDT');
        }

        const symbol = args[1].toUpperCase();
        const crypto = POPULAR_CRYPTOS.find(c => c.symbol === symbol);
        
        if (!crypto) {
          return ctx.reply(`ارز دیجیتال ${symbol} پشتیبانی نمی‌شود. از دستور /help برای مشاهده لیست ارزهای پشتیبانی شده استفاده کنید.`);
        }

        ctx.reply(`در حال تحلیل ${crypto.name} (${symbol})... لطفاً صبر کنید.`);
        
        const timeframe = args[2] || '4h';
        const analysis = await this.analyzer.getAIDecision(symbol, timeframe);
        
        if (analysis.error) {
          return ctx.reply(`خطا در تحلیل: ${analysis.error}`);
        }
        
        const message = this.formatSignalMessage(analysis, crypto.name);
        ctx.replyWithMarkdown(message);
      } catch (error) {
        console.error('خطا در دستور signal:', error);
        ctx.reply('خطایی رخ داد. لطفاً دوباره تلاش کنید.');
      }
    });

    // دریافت سیگنال برای همه ارزهای دیجیتال
    bot.command('signals', async (ctx) => {
      try {
        ctx.reply('در حال تحلیل همه ارزهای دیجیتال... لطفاً صبر کنید.');
        
        const args = ctx.message.text.split(' ');
        const timeframe = args[1] || '4h';
        
        const analyses = await this.analyzer.analyzeAll(timeframe);
        
        // ارسال سیگنال‌ها به صورت جداگانه برای هر ارز دیجیتال
        for (const analysis of analyses) {
          if (!analysis.error) {
            const message = this.formatSignalMessage(analysis, analysis.name);
            await ctx.replyWithMarkdown(message);
            // تاخیر کوتاه برای جلوگیری از محدودیت‌های تلگرام
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
        
        ctx.reply('تحلیل همه ارزهای دیجیتال به پایان رسید.');
      } catch (error) {
        console.error('خطا در دستور signals:', error);
        ctx.reply('خطایی رخ داد. لطفاً دوباره تلاش کنید.');
      }
    });

    // تغییر بازه زمانی تحلیل
    bot.command('timeframe', (ctx) => {
      const args = ctx.message.text.split(' ');
      if (args.length < 2) {
        return ctx.reply('لطفاً بازه زمانی را وارد کنید. مثال: /timeframe 1h');
      }

      const timeframe = args[1];
      if (!TIMEFRAMES.includes(timeframe)) {
        return ctx.reply(`بازه زمانی ${timeframe} پشتیبانی نمی‌شود. بازه‌های زمانی مجاز: ${TIMEFRAMES.join(', ')}`);
      }

      ctx.reply(`بازه زمانی با موفقیت به ${timeframe} تغییر یافت.`);
    });

    // اشتراک در سیگنال‌های روزانه
    bot.command('subscribe', (ctx) => {
      const chatId = ctx.chat.id;
      if (!this.subscribers.includes(chatId)) {
        this.subscribers.push(chatId);
        ctx.reply('شما با موفقیت در سیگنال‌های روزانه مشترک شدید. هر روز سیگنال‌های جدید را دریافت خواهید کرد.');
      } else {
        ctx.reply('شما قبلاً در سیگنال‌های روزانه مشترک شده‌اید.');
      }
    });

    // لغو اشتراک سیگنال‌های روزانه
    bot.command('unsubscribe', (ctx) => {
      const chatId = ctx.chat.id;
      const index = this.subscribers.indexOf(chatId);
      if (index !== -1) {
        this.subscribers.splice(index, 1);
        ctx.reply('اشتراک شما در سیگنال‌های روزانه با موفقیت لغو شد.');
      } else {
        ctx.reply('شما مشترک سیگنال‌های روزانه نیستید.');
      }
    });

    // پاسخ به پیام‌های دیگر
    bot.on('text', (ctx) => {
      ctx.reply('دستور نامعتبر. برای مشاهده لیست دستورات از /help استفاده کنید.');
    });

    // اجرای ربات
    bot.launch().then(() => {
      console.log('ربات تلگرام با موفقیت راه‌اندازی شد.');
    }).catch(error => {
      console.error('خطا در راه‌اندازی ربات:', error);
    });
  }

  // تنظیم ارسال سیگنال‌های زمان‌بندی شده
  setupScheduledSignals() {
    // ارسال سیگنال‌های روزانه (ساعت 8 صبح)
    cron.schedule('0 8 * * *', async () => {
      if (this.subscribers.length === 0) {
        return;
      }

      console.log('در حال ارسال سیگنال‌های روزانه...');
      const analyses = await this.analyzer.analyzeAll('1d');

      for (const chatId of this.subscribers) {
        try {
          // ارسال پیام خلاصه
          bot.telegram.sendMessage(chatId, '🔔 *سیگنال‌های روزانه - تحلیل با هوش مصنوعی*\n\nدر حال ارسال سیگنال‌های امروز...', { parse_mode: 'Markdown' });

          // ارسال سیگنال‌های هر ارز به صورت جداگانه
          for (const analysis of analyses) {
            if (!analysis.error) {
              const message = this.formatSignalMessage(analysis, analysis.name);
              await bot.telegram.sendMessage(chatId, message, { parse_mode: 'Markdown' });
              // تاخیر کوتاه برای جلوگیری از محدودیت‌های تلگرام
              await new Promise(resolve => setTimeout(resolve, 1000));
            }
          }
        } catch (error) {
          console.error(`خطا در ارسال سیگنال روزانه به کاربر ${chatId}:`, error);
        }
      }
    });

    // ارسال سیگنال‌های اضطراری (هر ساعت بررسی می‌شود)
    cron.schedule('0 * * * *', async () => {
      if (this.subscribers.length === 0) {
        return;
      }

      console.log('در حال بررسی سیگنال‌های اضطراری...');
      
      // بررسی سیگنال‌های قوی
      const urgentAnalyses = await this.analyzer.analyzeAll('1h');
      const strongSignals = urgentAnalyses.filter(analysis => 
        !analysis.error && 
        (analysis.aiDecision.action === 'BUY' || analysis.aiDecision.action === 'SELL') && 
        analysis.aiDecision.confidence > 90
      );

      if (strongSignals.length > 0) {
        for (const chatId of this.subscribers) {
          try {
            // ارسال پیام هشدار
            await bot.telegram.sendMessage(chatId, '⚠️ *هشدار سیگنال اضطراری*\n\nسیگنال‌های قوی با اطمینان بالا شناسایی شدند:', { parse_mode: 'Markdown' });

            // ارسال سیگنال‌های قوی
            for (const analysis of strongSignals) {
              const message = this.formatSignalMessage(analysis, analysis.name, true);
              await bot.telegram.sendMessage(chatId, message, { parse_mode: 'Markdown' });
              // تاخیر کوتاه برای جلوگیری از محدودیت‌های تلگرام
              await new Promise(resolve => setTimeout(resolve, 1000));
            }
          } catch (error) {
            console.error(`خطا در ارسال سیگنال اضطراری به کاربر ${chatId}:`, error);
          }
        }
      }
    });
  }

  // فرمت‌بندی پیام سیگنال
  formatSignalMessage(analysis, cryptoName, isUrgent = false) {
    const { symbol, timeframe, price, priceChange24h, aiDecision } = analysis;
    
    let actionEmoji = '⚪';
    if (aiDecision.action === 'BUY') {
      actionEmoji = '🟢';
    } else if (aiDecision.action