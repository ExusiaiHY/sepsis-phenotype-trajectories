#!/bin/bash
# ICU Sepsis Patient Browser - 快速启动脚本

cd "$(dirname "$0")/.."

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在"
    exit 1
fi

source .venv/bin/activate

echo "======================================================================"
echo "  ICU脓毒症科研患者浏览中台"
echo "  Patient Browser Medical BI Dashboard"
echo "======================================================================"
echo ""

# 检查端口占用
PORT=5051
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "⚠️  端口 $PORT 被占用，正在释放..."
    kill -9 $PID 2>/dev/null
    sleep 1
fi

echo "🚀 启动服务中..."
nohup python webapp_v2/app.py > webapp_v2/server.log 2>&1 &

sleep 2

# 检查服务是否启动
if curl -s http://localhost:$PORT/ > /dev/null; then
    echo ""
    echo "✅ 服务启动成功!"
    echo ""
    echo "🌐 访问地址:"
    echo "   http://localhost:$PORT"
    echo ""
    echo "📋 核心功能:"
    echo "   🏥 患者总览浏览 - 扁平化卡片列表"
    echo "   📊 科研数据看板 - 统计分析"
    echo "   🤖 AI智能分析   - 风险预测"
    echo "   ⚙️ 系统配置管理 - 系统设置"
    echo ""
    echo "📝 日志文件: webapp_v2/server.log"
    echo ""
    echo "⏹️  停止服务: lsof -ti:$PORT | xargs kill -9"
    echo "======================================================================"
    
    # 尝试自动打开浏览器
    if command -v open > /dev/null; then
        open http://localhost:$PORT
    elif command -v xdg-open > /dev/null; then
        xdg-open http://localhost:$PORT
    fi
else
    echo "❌ 服务启动失败，请检查日志"
    cat webapp_v2/server.log
fi
