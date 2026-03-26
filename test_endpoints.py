#!/usr/bin/env python3
"""
Script de testes para os novos endpoints de integração com Google Drive.

Testa:
1. POST /video-on-drive
2. POST /plate-code
3. GET /video-processing-status
"""

import requests
import json
from datetime import datetime
from typing import Optional

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

# Cores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    """Print mensagem de sucesso"""
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg: str):
    """Print mensagem de erro"""
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    """Print mensagem informativa"""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    """Print header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{msg}")
    print(f"{'='*70}{Colors.RESET}\n")

# ============================================================================
# TESTES
# ============================================================================

def test_video_on_drive():
    """
    Test 1: POST /video-on-drive
    Simula notificação do backend que um vídeo foi enviado ao Drive
    """
    print_header("TESTE 1: POST /video-on-drive")
    
    endpoint = f"{BASE_URL}/video-on-drive"
    
    # Payload de exemplo
    payload = {
        "drive_file_id": "1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": "ESP32_DEVICE_01",
            "noise_level": 95.5
        }
    }
    
    print(f"URL: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Requisição bem-sucedida!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            return True, result
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print_error(f"Erro de conexão: {BASE_URL} não está respondendo")
        print_info("Certifique-se que o servidor está rodando:")
        print_info("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False, None
    except Exception as e:
        print_error(f"Erro: {e}")
        return False, None


def test_plate_code():
    """
    Test 2: POST /plate-code
    Simula envio de código de placa do serviço de IA para backend
    """
    print_header("TESTE 2: POST /plate-code")
    
    endpoint = f"{BASE_URL}/plate-code"
    
    # Payload de exemplo
    payload = {
        "plate_code": "ABC1234",
        "confidence": 0.95,
        "video_id": "1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p",
        "additional_info": {
            "frame_detected": 150,
            "total_frames_analyzed": 280,
            "processing_time_seconds": 245.5
        }
    }
    
    print(f"URL: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Requisição bem-sucedida!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            return True, result
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False, None
            
    except Exception as e:
        print_error(f"Erro: {e}")
        return False, None


def test_video_processing_status(video_id: str):
    """
    Test 3: GET /video-processing-status/{video_id}
    Verifica status de processamento de um vídeo
    """
    print_header(f"TESTE 3: GET /video-processing-status/{video_id}")
    
    endpoint = f"{BASE_URL}/video-processing-status/{video_id}"
    
    print(f"URL: {endpoint}\n")
    
    try:
        response = requests.get(
            endpoint,
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Status obtido com sucesso!")
            
            print(f"\nDetalhes:")
            print(f"  Status: {result.get('status', 'N/A')}")
            print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
            
            if 'result' in result:
                print(f"  Resultado: {json.dumps(result['result'], indent=2)}")
            if 'error' in result:
                print(f"  Erro: {result['error']}")
            
            return True, result
        
        elif response.status_code == 404:
            print_info(f"Vídeo '{video_id}' não encontrado na fila de processamento")
            return False, None
        
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False, None
            
    except Exception as e:
        print_error(f"Erro: {e}")
        return False, None


def test_all_video_statuses():
    """
    Test 4: GET /video-processing-status
    Lista status de todos os vídeos em processamento
    """
    print_header("TESTE 4: GET /video-processing-status (Todos os vídeos)")
    
    endpoint = f"{BASE_URL}/video-processing-status"
    
    print(f"URL: {endpoint}\n")
    
    try:
        response = requests.get(
            endpoint,
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Status obtido com sucesso!")
            
            total = result.get('total_videos', 0)
            print(f"\nTotal de vídeos em processamento: {total}")
            
            if total > 0:
                print(f"\nDetalhes:")
                videos = result.get('videos', {})
                for video_id, status in videos.items():
                    print(f"\n  Video ID: {video_id}")
                    print(f"    Status: {status.get('status', 'N/A')}")
                    if 'error' in status:
                        print(f"    Erro: {status['error']}")
            
            return True, result
        
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False, None
            
    except Exception as e:
        print_error(f"Erro: {e}")
        return False, None


def test_report_infraction():
    """
    Test 5: GET /dashboard/data
    Retorna dados do dashboard (endpoint existente, para referência)
    """
    print_header("TESTE 5: GET /dashboard/data (Endpoint Existente)")
    
    endpoint = f"{BASE_URL}/dashboard/data"
    
    print(f"URL: {endpoint}\n")
    
    try:
        response = requests.get(
            endpoint,
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print_success("Dados obtidos com sucesso!")
            
            if isinstance(result, list):
                print(f"Total de registros: {len(result)}")
                if len(result) > 0:
                    print(f"\nPrimeiro registro:")
                    print(f"{json.dumps(result[0], indent=2)}")
            
            return True, result
        
        else:
            print_error(f"Status {response.status_code}: {response.text}")
            return False, None
            
    except Exception as e:
        print_error(f"Erro: {e}")
        return False, None


def test_full_workflow():
    """
    Test 6: Fluxo completo simulado
    Simula o fluxo inteiro: notificação → aguarda status → envia placa
    """
    print_header("TESTE 6: Fluxo Completo Simulado")
    
    print("Executando sequência de testes que simula o fluxo real:\n")
    
    # Passo 1
    print(f"{Colors.BOLD}Passo 1/3: Notificar vídeo no Drive{Colors.RESET}")
    success, result_1 = test_video_on_drive()
    
    if not success:
        print_error("Fluxo interrompido - Passo 1 falhou")
        return False
    
    video_id = result_1.get('drive_file_id')
    
    # Passo 2
    print(f"\n{Colors.BOLD}Passo 2/3: Verificar status (pode estar processando){Colors.RESET}")
    success, result_2 = test_video_processing_status(video_id)
    
    if not success:
        print_info("Status não disponível ainda (é esperado se Google Drive não estiver configurado)")
    
    # Passo 3
    print(f"\n{Colors.BOLD}Passo 3/3: Enviar código de placa{Colors.RESET}")
    success, result_3 = test_plate_code()
    
    if success:
        print_success("✅ Fluxo completo executado com sucesso!")
        return True
    else:
        print_error("❌ Fluxo interrompido")
        return False


# ============================================================================
# MENU PRINCIPAL
# ============================================================================

def main():
    """Menu principal de testes"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  TESTES - Motorcycle Noise Detector (Google Drive Integration)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}URL do Servidor: {BASE_URL}{Colors.RESET}\n")
    
    print("Escolha uma opção:")
    print("1 - Teste individual: POST /video-on-drive")
    print("2 - Teste individual: POST /plate-code")
    print("3 - Teste individual: GET /video-processing-status/{id}")
    print("4 - Teste individual: GET /video-processing-status")
    print("5 - Teste individual: GET /dashboard/data")
    print("6 - Teste fluxo completo (recomendado)")
    print("7 - Todos os testes")
    print("0 - Sair\n")
    
    choice = input("Digite a opção (0-7): ").strip()
    
    if choice == "1":
        test_video_on_drive()
    elif choice == "2":
        test_plate_code()
    elif choice == "3":
        video_id = input("Digite o Video ID (Google Drive file ID): ").strip()
        if video_id:
            test_video_processing_status(video_id)
        else:
            print_error("Video ID inválido")
    elif choice == "4":
        test_all_video_statuses()
    elif choice == "5":
        test_report_infraction()
    elif choice == "6":
        test_full_workflow()
    elif choice == "7":
        print("\nExecutando todos os testes...\n")
        test_video_on_drive()
        test_plate_code()
        test_all_video_statuses()
        test_report_infraction()
        test_full_workflow()
    elif choice == "0":
        print("Encerrando...")
        return
    else:
        print_error("Opção inválida")
    
    # Repetir menu
    input("\n\nPressione ENTER para continuar...")
    main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Programa interrompido pelo usuário{Colors.RESET}\n")
    except Exception as e:
        print_error(f"Erro inesperado: {e}")
