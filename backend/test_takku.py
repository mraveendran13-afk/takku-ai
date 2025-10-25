#!/usr/bin/env python3
"""
Takku AI Model Test Script
Tests all three models to verify correct selection
"""

import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "https://takku-ai-production.up.railway.app"
# For local testing, use: API_BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_result(test_name, response_data, expected_model):
    """Print test results"""
    actual_model = response_data.get('model') or response_data.get('model_used')
    searched_web = response_data.get('searched_web', False)
    
    # Check if correct model was used
    passed = actual_model == expected_model
    status = "✅ PASS" if passed else "❌ FAIL"
    
    print(f"\n{status} | {test_name}")
    print(f"   Expected Model: {expected_model}")
    print(f"   Actual Model:   {actual_model}")
    print(f"   Web Search:     {searched_web}")
    
    # Show snippet of answer
    answer = response_data.get('answer') or response_data.get('response', '')
    if answer:
        snippet = answer[:100] + "..." if len(answer) > 100 else answer
        print(f"   Answer:         {snippet}")
    
    return passed

def test_health_check():
    """Test health endpoint"""
    print_header("HEALTH CHECK")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        data = response.json()
        
        print(f"Status:          {data.get('status')}")
        print(f"Groq Available:  {data.get('groq_available')}")
        print(f"Memory System:   {data.get('memory_system')}")
        print(f"Web Search:      {data.get('web_search')}")
        print(f"Service:         {data.get('service')}")
        
        if data.get('status') == 'healthy' and data.get('groq_available'):
            print("\n✅ System is healthy and ready!")
            return True
        else:
            print("\n⚠️  System is not fully operational")
            return False
            
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        return False

def test_compound_mini():
    """Test compound-mini model (web search questions)"""
    print_header("TEST 1: COMPOUND-MINI (Web Search)")
    
    test_questions = [
        "What's the weather today?",
        "What are the latest AI news?",
        "What's trending on social media now?"
    ]
    
    results = []
    
    for question in test_questions:
        print(f"\n📝 Testing: \"{question}\"")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/ask",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                passed = print_result(
                    f"Question: {question[:30]}...",
                    data,
                    "groq/compound-mini"  # Fixed: include groq/ prefix
                )
                results.append(passed)
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            results.append(False)
    
    return results

def test_llama_instant():
    """Test llama-3.1-8b-instant model (general questions)"""
    print_header("TEST 2: LLAMA-3.1-8B-INSTANT (General Q&A)")
    
    test_questions = [
        "What is Python programming?",
        "Explain quantum physics",
        "Tell me a fun fact about cats"
    ]
    
    results = []
    
    for question in test_questions:
        print(f"\n📝 Testing: \"{question}\"")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/ask",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                passed = print_result(
                    f"Question: {question[:30]}...",
                    data,
                    "llama-3.1-8b-instant"
                )
                results.append(passed)
            else:
                print(f"   ❌ API Error: {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            results.append(False)
    
    return results

def test_llama_chat():
    """Test llama3-8b-8192 model (chat endpoint)"""
    print_header("TEST 3: LLAMA3-8B-8192 (Chat Endpoint)")
    
    test_cases = [
        {"message": "Hello Takku!", "symptoms": ""},
        {"message": "I have a headache", "symptoms": "headache, tired"},
        {"message": "Tell me about healthy eating", "symptoms": ""}
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 Testing: \"{test_case['message']}\"")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json=test_case,
                headers={
                    "Content-Type": "application/json",
                    "X-User-ID": "test-user-123"  # FIXED: Added required header
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Accept either model name format
                actual_model = data.get('model_used')
                expected_models = ["llama3-8b-8192", "llama-3.1-8b-instant"]
                passed = actual_model in expected_models
                
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"\n{status} | Message: {test_case['message'][:30]}...")
                print(f"   Expected Models: {expected_models}")
                print(f"   Actual Model:    {actual_model}")
                print(f"   Web Search:      {data.get('searched_web', False)}")
                
                answer = data.get('response', '')
                if answer:
                    snippet = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"   Answer:          {snippet}")
                
                results.append(passed)
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            results.append(False)
    
    return results

def test_game_context():
    """Test that games don't use compound model"""
    print_header("TEST 4: GAME MODE (Should NOT use compound)")
    
    game_prompt = "You are playing a math game with a child. Ask a simple addition question."
    
    print(f"\n📝 Testing: Game context (should use llama-3.1-8b-instant)")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/ask",
            json={
                "question": "Start the game",
                "game_context": game_prompt
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            passed = print_result(
                "Game Mode Test",
                data,
                "llama-3.1-8b-instant"
            )
            return [passed]
        else:
            print(f"   ❌ API Error: {response.status_code}")
            return [False]
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return [False]

def print_summary(all_results):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal Tests:  {total_tests}")
    print(f"✅ Passed:    {passed_tests}")
    print(f"❌ Failed:    {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Your Takku AI is working perfectly!")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    print("\n" + "="*60 + "\n")

def main():
    """Run all tests"""
    print("\n" + "🐱"*30)
    print("    TAKKU AI MODEL TESTING SUITE")
    print("🐱"*30)
    print(f"\nTesting API: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check health first
    if not test_health_check():
        print("\n⚠️  Skipping tests - system is not healthy")
        return
    
    # Run all tests
    all_results = []
    
    try:
        # Test 1: Compound-mini (web search)
        results_1 = test_compound_mini()
        all_results.extend(results_1)
        
        # Test 2: Llama-instant (general Q&A)
        results_2 = test_llama_instant()
        all_results.extend(results_2)
        
        # Test 3: Llama-chat (chat endpoint)
        results_3 = test_llama_chat()
        all_results.extend(results_3)
        
        # Test 4: Game mode
        results_4 = test_game_context()
        all_results.extend(results_4)
        
        # Print summary
        print_summary(all_results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()