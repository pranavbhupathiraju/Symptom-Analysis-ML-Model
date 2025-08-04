#!/bin/bash

# LifeSaver API Runner Script

echo "🚑 LifeSaver API - Emergency Health Triage"
echo "=========================================="

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "❌ Maven is not installed. Please install Maven first:"
    echo "   - macOS: brew install maven"
    echo "   - Ubuntu/Debian: sudo apt install maven"
    echo "   - Windows: Download from https://maven.apache.org/download.cgi"
    exit 1
fi

# Check if Java 17+ is available
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 17 ]; then
    echo "❌ Java 17 or higher is required. Current version: $JAVA_VERSION"
    echo "   Please upgrade your Java version."
    exit 1
fi

echo "✅ Java version: $(java -version 2>&1 | head -n 1)"
echo "✅ Maven version: $(mvn -version | head -n 1)"

echo ""
echo "🔧 Building the project..."
mvn clean compile

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting LifeSaver API..."
    echo "   - API will be available at: http://localhost:8080"
    echo "   - Expected Flask ML service at: http://localhost:5000"
    echo ""
    echo "📋 Available endpoints:"
    echo "   - GET  /api/v1/symptoms"
    echo "   - POST /api/v1/triage"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    mvn spring-boot:run
else
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi 