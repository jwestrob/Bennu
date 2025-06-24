#!/bin/bash
set -euo pipefail

# Code Interpreter Deployment Script
# Provides secure deployment options with different security levels

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="code-interpreter"
GVISOR_SERVICE_NAME="code-interpreter-gvisor"
PORT=8000

print_header() {
    echo -e "${BLUE}==== Code Interpreter Deployment ====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    echo "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check for Docker Compose (both V1 and V2)
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_success "Dependencies satisfied (using $DOCKER_COMPOSE_CMD)"
}

check_gvisor() {
    if command -v runsc &> /dev/null; then
        print_success "gVisor (runsc) is available"
        return 0
    else
        print_warning "gVisor (runsc) is not installed"
        return 1
    fi
}

build_image() {
    echo "Building code interpreter Docker image..."
    docker build -t code-interpreter:latest .
    print_success "Docker image built successfully"
}

deploy_standard() {
    echo "Deploying with standard Docker security..."
    $DOCKER_COMPOSE_CMD up -d $SERVICE_NAME
    print_success "Service deployed on port $PORT"
    print_warning "Running with standard Docker security (not maximum security)"
}

deploy_gvisor() {
    echo "Deploying with gVisor maximum security..."
    $DOCKER_COMPOSE_CMD --profile gvisor up -d $GVISOR_SERVICE_NAME
    print_success "Service deployed with gVisor on port $PORT"
    print_success "Maximum security configuration active"
}

check_health() {
    echo "Waiting for service to become healthy..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
            print_success "Service is healthy!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "Service failed to become healthy within $(($max_attempts * 2)) seconds"
    return 1
}

show_status() {
    echo -e "\n${BLUE}Service Status:${NC}"
    $DOCKER_COMPOSE_CMD ps
    
    echo -e "\n${BLUE}Service Logs (last 20 lines):${NC}"
    $DOCKER_COMPOSE_CMD logs --tail=20 $SERVICE_NAME 2>/dev/null || \
    $DOCKER_COMPOSE_CMD logs --tail=20 $GVISOR_SERVICE_NAME 2>/dev/null || \
    echo "No logs available"
}

stop_services() {
    echo "Stopping code interpreter services..."
    $DOCKER_COMPOSE_CMD down
    print_success "Services stopped"
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  deploy     Deploy with standard Docker security"
    echo "  deploy-max Deploy with maximum security (gVisor)"
    echo "  status     Show service status and logs"
    echo "  stop       Stop all services"
    echo "  restart    Restart services"
    echo "  logs       Show service logs"
    echo "  test       Run a test execution"
    echo ""
    echo "Security Levels:"
    echo "  standard   - Docker security features (capabilities, read-only, etc.)"
    echo "  maximum    - gVisor + all Docker security features"
}

run_test() {
    echo "Running test execution..."
    
    # Test basic functionality
    local test_code="print('Code interpreter is working!')"
    local response=$(curl -s -X POST "http://localhost:$PORT/execute" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\": \"test-session\", \"code\": \"$test_code\", \"timeout\": 10}")
    
    if echo "$response" | grep -q '"success": true'; then
        print_success "Test execution successful"
        echo "Response: $response"
    else
        print_error "Test execution failed"
        echo "Response: $response"
        return 1
    fi
}

main() {
    print_header
    
    case "${1:-}" in
        "build")
            check_dependencies
            build_image
            ;;
        "deploy")
            check_dependencies
            build_image
            deploy_standard
            check_health
            show_status
            ;;
        "deploy-max")
            check_dependencies
            if check_gvisor; then
                build_image
                deploy_gvisor
                check_health
                show_status
            else
                print_error "gVisor is required for maximum security deployment"
                echo "Install gVisor: https://gvisor.dev/docs/user_guide/install/"
                exit 1
            fi
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            deploy_standard
            check_health
            ;;
        "logs")
            $DOCKER_COMPOSE_CMD logs -f $SERVICE_NAME 2>/dev/null || \
            $DOCKER_COMPOSE_CMD logs -f $GVISOR_SERVICE_NAME 2>/dev/null
            ;;
        "test")
            run_test
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

main "$@"