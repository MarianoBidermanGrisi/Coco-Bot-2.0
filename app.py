# app.py - Bot de trading como Web Service (con recarga completa cada 45s)
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import time
import json
import os
import asyncio
from flask import Flask, render_template_string
from threading import Thread
import logging
import sys

# ===============================
# 🔐 Configuración del Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# 🔐 Configuración Binance
# ===============================
BINANCE_API_KEY = "oCYyOTBPPLr2ggLx8yszPRjSWxEecNQIL7U2iFPyhDTwsXNcD3otGMo1FtOdotHA"
BINANCE_API_SECRET = "9qtqNGYJSQqJPVQPaRLt0vYbRo4IPnSj3hby1sRUoWBbhqI4ETfRvsNPHyyZbflx"

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Símbolos y timeframes
SYMBOLS = ["DOGEUSDT", "XRPUSDT", "ETHUSDT", "AVAXUSDT", "TRXUSDT", "XLMUSDT", "SOLUSDT"]
TIMEFRAMES = ["1m", "3m", "5m", "15m"]
DATA_LIMIT = 1000
MEMORIA_FILE = "memoria_senales.json"
EVALUACION_MINUTOS = 30
UMBRAL_CONFLUENCIA = 3.0  # 🔥 Más sensibilidad = más señales original 6.0

# ===============================
# 🛠️ Funciones técnicas
# ===============================
def sma(s, l): return s.rolling(l).mean()
def ema(s, l): return s.ewm(span=l, adjust=False).mean()
def stdev(s, l): return s.rolling(l).std()
def change(s): return s.diff()
def sign(s): return np.sign(s)
def crossover(s1, s2): return (s1 > s2) & (s1.shift(1) <= s2.shift(1))
def crossunder(s1, s2): return (s1 < s2) & (s1.shift(1) >= s2.shift(1))

# ===============================
# 📥 Descargar datos
# ===============================
def descargar_datos(symbol: str, interval: str, limit: int):
    for intento in range(3):
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            if not df.empty and len(df) >= 200:
                return df
        except Exception as e:
            logger.info(f"❌ Error descargando {symbol} ({interval}) - Intento {intento + 1}: {e}")
            time.sleep(2)
    logger.info(f"⚠️ {symbol} ({interval}): No se pudieron obtener datos")
    return None

# ===============================
# 🧠 Calcular puntaje de señal
# ===============================
def calcular_puntaje_senal(df: pd.DataFrame, pesos: dict) -> dict:
    if len(df) < 200 or df['close'].isna().sum() > len(df) * 0.1:
        return {"compra": 0, "venta": 0}

    window_len = 28
    v_len = 14

    price_spread = stdev(df['high'] - df['low'], window_len)
    signed_volume = sign(change(df['close'])) * df['volume']
    v = signed_volume.cumsum()
    smooth = sma(v, v_len)
    v_spread = stdev(v - smooth, window_len)
    shadow = ((v - smooth) / v_spread) * price_spread
    out = np.where(shadow > 0, df['high'] + shadow, df['low'] + shadow)
    obvema = ema(pd.Series(out), 1).ffill().bfill()

    if obvema.isna().all():
        return {"compra": 0, "venta": 0}

    def dema(s, l):
        ma1 = ema(s, l)
        ma2 = ema(ma1, l)
        return 2 * ma1 - ma2

    ma = dema(obvema, 9)
    slow_ma = ema(df['close'], 26)
    macd_line = ma - slow_ma

    if macd_line.isna().sum() > len(macd_line) * 0.5:
        return {"compra": 0, "venta": 0}

    def calc_slope(series, length):
        slope_list = []
        for i in range(len(series)):
            if i < length:
                slope_list.append(np.nan)
                continue
            y = series.iloc[i-length:i].values
            x = np.arange(1, length+1)
            valid_mask = ~np.isnan(y)
            if not np.any(valid_mask):
                slope_list.append(np.nan)
                continue
            y_valid = y[valid_mask]
            x_valid = x[valid_mask]
            if len(x_valid) < 2:
                slope_list.append(np.nan)
                continue
            A = np.vstack([x_valid, np.ones(len(x_valid))]).T
            m, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
            slope_list.append(m)
        return pd.Series(slope_list, index=series.index)

    tt1 = calc_slope(macd_line, 2).reindex(df.index).ffill().bfill()
    if tt1.isna().all():
        return {"compra": 0, "venta": 0}

    diff_tt1 = (tt1 - tt1.shift(1)).abs()
    cumSum = diff_tt1.cumsum()
    n = np.arange(1, len(cumSum) + 1)
    a15 = cumSum / n * 1.0

    b5 = pd.Series(index=df.index, dtype=float)
    oc = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        idx = df.index[i]
        if i == 0:
            b5[idx] = tt1[idx]
            oc[idx] = 0
        else:
            prev_idx = df.index[i - 1]
            prev_b5 = b5[prev_idx]
            current_a15 = a15[idx]
            if tt1[idx] > prev_b5 + current_a15:
                b5[idx] = tt1[idx]
            elif tt1[idx] < prev_b5 - current_a15:
                b5[idx] = tt1[idx]
            else:
                b5[idx] = prev_b5
            oc[idx] = 1 if b5[idx] > prev_b5 else (-1 if b5[idx] < prev_b5 else oc[prev_idx])

    src4 = df['close']
    l30 = 6
    k30 = 1.0 / l30
    vma_val = pd.Series(0.0, index=df.index)
    iS = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i - 1]
        pdm = max(src4[idx] - src4[prev_idx], 0)
        mdm = max(src4[prev_idx] - src4[idx], 0)
        pdmS = (1-k30)*(vma_val[prev_idx] if i>1 else 0) + k30*pdm
        mdmS = (1-k30)*0 + k30*mdm
        s_val = pdmS + mdmS
        pdi = pdmS/s_val if s_val else 0
        mdi = mdmS/s_val if s_val else 0
        pdiS = (1-k30)*0 + k30*pdi
        mdiS = (1-k30)*0 + k30*mdi
        d = abs(pdiS - mdiS)
        s1 = pdiS + mdiS
        iS_val = d/s1 if s1 else 0
        iS[idx] = (1-k30)*iS[prev_idx] + k30*iS_val
        start_idx = max(0, i-l30+1)
        window = iS.iloc[start_idx:i+1]
        hhv = window.max()
        llv = window.min()
        vI = (iS[idx] - llv)/(hhv - llv) if hhv != llv else 0
        vma_val[idx] = (1-k30*vI)*vma_val[prev_idx] + k30*vI*src4[idx]

    df['vma'] = vma_val

    h = 8.0
    mult = 3.0
    coefs = np.array([np.exp(-(i**2)/(2*h*h)) for i in range(50)])
    den = coefs.sum()
    out30 = df['close'].rolling(50).apply(lambda x: (x * coefs[::-1]).sum() / den, raw=True)
    mae = (df['close'] - out30).abs().rolling(50).mean() * mult
    df['nwe_upper'] = out30 + mae
    df['nwe_lower'] = out30 - mae

    basis = sma(df['close'], 200)
    dev = 2 * stdev(df['close'], 200)
    df['BBupper'] = basis + dev
    df['BBlower'] = basis - dev

    ult = df.iloc[-1]
    antepenult = df.iloc[-3]

    puntos_compra = 0
    puntos_venta = 0

    cond_cocodrilo_up = (
        (df['close'].iloc[-3] > df['open'].iloc[-3]) and
        (df['close'].iloc[-2] > df['open'].iloc[-2]) and
        (df['close'].iloc[-1] > df['open'].iloc[-1]) and
        (antepenult['low'] <= antepenult['BBlower']) and
        (ult['vma'] > df['vma'].iloc[-2])
    )
    cond_showsignal_up = (oc.iloc[-1] == 1) and (oc.iloc[-2] in [0, -1])
    cond_nwe_crossover = crossunder(df['close'], df['nwe_lower']).iloc[-1]
    cond_vma_up = ult['vma'] > df['vma'].iloc[-2]

    puntos_compra = sum([
        cond_cocodrilo_up * pesos["cocodriloup"],
        cond_showsignal_up * pesos["showsignal_up"],
        cond_nwe_crossover * pesos["nwe_crossover"],
        cond_vma_up * pesos["vma_trend_up"]
    ])

    cond_cocodrilo_dn = (
        (df['close'].iloc[-3] < df['open'].iloc[-3]) and
        (df['close'].iloc[-2] < df['open'].iloc[-2]) and
        (df['close'].iloc[-1] < df['open'].iloc[-1]) and
        (antepenult['high'] >= antepenult['BBupper']) and
        (ult['vma'] < df['vma'].iloc[-2])
    )
    cond_showsignal_dn = (oc.iloc[-1] == -1) and (oc.iloc[-2] in [0, 1])
    cond_nwe_crossunder = crossover(df['close'], df['nwe_upper']).iloc[-1]
    cond_vma_dn = ult['vma'] < df['vma'].iloc[-2]

    puntos_venta = sum([
        cond_cocodrilo_dn * pesos["cocodrilodn"],
        cond_showsignal_dn * pesos["showsignal_down"],
        cond_nwe_crossunder * pesos["nwe_crossunder"],
        cond_vma_dn * pesos["vma_trend_down"]
    ])

    return {"compra": puntos_compra, "venta": puntos_venta}

# ===============================
# 💬 Telegram Setup
# ===============================
from telegram import Bot

TELEGRAM_TOKEN = "7969091726:AAFVTZAlWN0aA6uMtJgWfnQhzTRD3cpx4wM"  # ← Reemplaza con tu token
TELEGRAM_CHAT_ID = 1570204748     # ← Reemplaza con tu ID

bot = Bot(token=TELEGRAM_TOKEN)

async def enviar_alerta_telegram(mensaje: str):
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=mensaje,
            parse_mode='Markdown'
        )
        logger.info("✅ Alerta enviada por Telegram")
    except Exception as e:
        logger.error(f"❌ Error enviando a Telegram: {e}")

# ===============================
# 💾 Memoria y aprendizaje
# ===============================
def cargar_memoria():
    if not os.path.exists(MEMORIA_FILE):
        return []
    try:
        with open(MEMORIA_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("⚠️ memoria_senales.json vacío o corrupto → creando nuevo")
        return []
    except Exception as e:
        logger.error(f"❌ Error al cargar memoria: {e}")
        return []

def guardar_memoria(memoria):
    with open(MEMORIA_FILE, "w") as f:
        json.dump(memoria, f, indent=2, default=str)

def evaluar_resultados(memoria):
    """Evalúa si las señales pasadas fueron acertadas"""
    ahora = datetime.now()
    for registro in memoria:
        if registro["resultado"] is not None:
            continue

        ts = datetime.fromisoformat(registro["timestamp"])
        if (ahora - ts).total_seconds() < EVALUACION_MINUTOS * 60:
            continue

        try:
            df = descargar_datos(registro["symbol"], "15m", 50)
            if df is None or len(df) < 10:
                continue

            precio_entrada = registro["precio_entrada"]
            ult_precio = df['close'].iloc[-1]

            if registro["tipo"] == "COMPRA":
                registro["resultado"] = "GANANCIA" if ult_precio > precio_entrada else "PÉRDIDA"
            elif registro["tipo"] == "VENTA":
                registro["resultado"] = "GANANCIA" if ult_precio < precio_entrada else "PÉRDIDA"

        except Exception as e:
            logger.error(f"❌ Error evaluando resultado: {e}")

    guardar_memoria(memoria)

def ajustar_pesos(memoria, pesos):
    """Ajusta pesos basado en precisión histórica"""
    aciertos = {
        "cocodriloup": {"ganancia": 0, "total": 0},
        "showsignal_up": {"ganancia": 0, "total": 0},
        "nwe_crossover": {"ganancia": 0, "total": 0},
        "vma_trend_up": {"ganancia": 0, "total": 0},
        "cocodrilodn": {"ganancia": 0, "total": 0},
        "showsignal_down": {"ganancia": 0, "total": 0},
        "nwe_crossunder": {"ganancia": 0, "total": 0},
        "vma_trend_down": {"ganancia": 0, "total": 0}
    }

    for reg in memoria[-100:]:
        tipo = reg["tipo"].lower()
        res = reg.get("resultado")
        if res is None:
            continue

        if tipo == "compra":
            aciertos["cocodriloup"]["total"] += 1
            aciertos["showsignal_up"]["total"] += 1
            aciertos["nwe_crossover"]["total"] += 1
            aciertos["vma_trend_up"]["total"] += 1
            if res == "GANANCIA":
                aciertos["cocodriloup"]["ganancia"] += 1
                aciertos["showsignal_up"]["ganancia"] += 1
                aciertos["nwe_crossover"]["ganancia"] += 1
                aciertos["vma_trend_up"]["ganancia"] += 1

        elif tipo == "venta":
            aciertos["cocodrilodn"]["total"] += 1
            aciertos["showsignal_down"]["total"] += 1
            aciertos["nwe_crossunder"]["total"] += 1
            aciertos["vma_trend_down"]["total"] += 1
            if res == "GANANCIA":
                aciertos["cocodrilodn"]["ganancia"] += 1
                aciertos["showsignal_down"]["ganancia"] += 1
                aciertos["nwe_crossunder"]["ganancia"] += 1
                aciertos["vma_trend_down"]["ganancia"] += 1

    for key in pesos:
        total = aciertos[key]["total"]
        ganancia = aciertos[key]["ganancia"]
        precision = ganancia / total if total > 0 else 0.5

        if precision < 0.5:
            pesos[key] *= 0.95  # Penalizar
        elif precision > 0.7:
            pesos[key] *= 1.05  # Reforzar

    logger.info("🧠 Pesos ajustados según rendimiento reciente")

# ===============================
# 🌐 Web Server (Flask)
# ===============================
app = Flask(__name__)

# Variable global para almacenar el último análisis
ultimo_analisis = {
    "fecha": "",
    "resultados": [],
    "mensaje": ""
}

@app.route('/')
def index():
    # Renderizamos la página completa con los últimos resultados
    resultados_html = ""
    for res in ultimo_analisis["resultados"]:
        clase = "log error" if "Sin señal" in res else "log success"
        resultados_html += f'<div class="{clase}">{res}</div>\n'

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Bot de Trading - Estado Actual</title>
        <meta http-equiv="refresh" content="45"> <!-- 🔁 Recarga cada 45 segundos -->
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #fff; }}
            .log {{ 
                background: #f8f9fa; 
                padding: 10px; 
                margin: 8px 0; 
                border-radius: 4px;
                border-left: 4px solid #007BFF; 
            }}
            .log.error {{ border-left-color: #dc3545; color: #721c24; }}
            .log.success {{ border-left-color: #28a745; color: #155724; }}
            h1 {{ color: #333; }}
            .fecha {{ font-weight: bold; color: #007bff; }}
        </style>
    </head>
    <body>
        <h1>🤖 Bot de Trading - Estado Actual</h1>
        <div class="log"><strong>Última ejecución:</strong> <span class="fecha">{ultimo_analisis["fecha"]}</span></div>
        {resultados_html}
        <div class="log"><strong>Mensaje:</strong> {ultimo_analisis["mensaje"] or "Esperando próxima señal..."}</div>
        <p><em>🔄 Esta página se recarga automáticamente cada 45 segundos.</em></p>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/health')
def health():
    """Health check para Render"""
    return {"status": "ok"}

# ===============================
# 🔁 Análisis principal
# ===============================
def ejecutar_analisis():
    global ultimo_analisis
    logger.info(f"\n📆 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] Iniciando análisis...")
    ultimo_analisis["fecha"] = datetime.now().strftime('%Y-%m-%d %H:%M')
    ultimo_analisis["resultados"] = []
    ultimo_analisis["mensaje"] = ""

    memoria = cargar_memoria()
    evaluar_resultados(memoria)

    pesos = {
        "cocodriloup": 2.0,
        "showsignal_up": 1.5,
        "nwe_crossover": 1.5,
        "vma_trend_up": 1.0,
        "cocodrilodn": 2.0,
        "showsignal_down": 1.5,
        "nwe_crossunder": 1.5,
        "vma_trend_down": 1.0
    }
    ajustar_pesos(memoria, pesos)

    for symbol in SYMBOLS:
        logger.info(f"🔍 Analizando {symbol}...")
        ultimo_analisis["resultados"].append(f"Analizando {symbol}...")
        puntaje_total_compra = 0
        puntaje_total_venta = 0
        detalles = []
        precio_actual = None

        for tf in TIMEFRAMES:
            df = descargar_datos(symbol, tf, DATA_LIMIT)
            if df is None or len(df) < 200:
                continue

            puntajes = calcular_puntaje_senal(df, pesos)
            precio_actual = df['close'].iloc[-1]
            puntaje_total_compra += puntajes["compra"]
            puntaje_total_venta += puntajes["venta"]

            detalles.append({
                "tf": tf,
                "compra": puntajes["compra"],
                "venta": puntajes["venta"],
                "precio": float(precio_actual)
            })

        if precio_actual is None:
            ultimo_analisis["resultados"].append(f"🟡 Sin señal clara para {symbol}")
            continue

        if puntaje_total_compra >= UMBRAL_CONFLUENCIA and puntaje_total_compra > puntaje_total_venta:
            stop_loss = precio_actual * 0.99
            mensaje = f"""
🟢 *SEÑAL DE COMPRA CONFLUENTE*
──────────────────────────────
*Par:* `{symbol}`
*Precio de entrada:* `${precio_actual:,.6f}`
*Stop Loss (1%):* `${stop_loss:,.6f}`
*Puntaje total:* `{puntaje_total_compra:.1f}`
*Timeframes activos:* `{[d['tf'] for d in detalles if d['compra'] > 0]}`
*Fecha:* {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            logger.info(mensaje.strip())
            ultimo_analisis["resultados"].append(f"✅ SEÑAL DE COMPRA en {symbol}")
            ultimo_analisis["mensaje"] = "Señal de compra detectada"

            asyncio.run(enviar_alerta_telegram(mensaje))

            memoria.append({
                "symbol": symbol,
                "tipo": "COMPRA",
                "precio_entrada": precio_actual,
                "stop_loss": stop_loss,
                "timeframes": [d['tf'] for d in detalles if d['compra'] > 0],
                "puntaje": puntaje_total_compra,
                "timestamp": datetime.now().isoformat(),
                "resultado": None
            })

        elif puntaje_total_venta >= UMBRAL_CONFLUENCIA and puntaje_total_venta > puntaje_total_compra:
            stop_loss = precio_actual * 1.01
            mensaje = f"""
🔴 *SEÑAL DE VENTA CONFLUENTE*
──────────────────────────────
*Par:* `{symbol}`
*Precio de entrada:* `${precio_actual:,.6f}`
*Stop Loss (1%):* `${stop_loss:,.6f}`
*Puntaje total:* `{puntaje_total_venta:.1f}`
*Timeframes activos:* `{[d['tf'] for d in detalles if d['venta'] > 0]}`
*Fecha:* {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            logger.info(mensaje.strip())
            ultimo_analisis["resultados"].append(f"✅ SEÑAL DE VENTA en {symbol}")
            ultimo_analisis["mensaje"] = "Señal de venta detectada"

            asyncio.run(enviar_alerta_telegram(mensaje))

            memoria.append({
                "symbol": symbol,
                "tipo": "VENTA",
                "precio_entrada": precio_actual,
                "stop_loss": stop_loss,
                "timeframes": [d['tf'] for d in detalles if d['venta'] > 0],
                "puntaje": puntaje_total_venta,
                "timestamp": datetime.now().isoformat(),
                "resultado": None
            })
        else:
            ultimo_analisis["resultados"].append(f"🟡 Sin señal clara para {symbol}")

    guardar_memoria(memoria)

# ===============================
# ▶️ Ejecución automática + Web Server
# ===============================
if __name__ == "__main__":
    # Crear archivo de memoria si no existe
    if not os.path.exists(MEMORIA_FILE):
        with open(MEMORIA_FILE, "w") as f:
            f.write("[]")

    # Forzar una primera ejecución
    ejecutar_analisis()

    # Iniciar el analizador en segundo plano
    def iniciar_analizador():
        while True:
            try:
                ejecutar_analisis()
            except Exception as e:
                logger.error(f"❌ Error en el analizador: {e}")
            time.sleep(45)  # Cada 45 segundos

    thread = Thread(target=iniciar_analizador)
    thread.start()

    # Iniciar el servidor web
    app.run(host='0.0.0.0', port=10000, debug=False)


