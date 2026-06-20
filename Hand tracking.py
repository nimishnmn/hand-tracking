#!/usr/bin/env python3
import os
import sys
import socket
import webbrowser
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 8000

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable browser caching so updates are immediately loaded
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def find_free_port(start_port=8000):
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                port += 1
    return port

def main():
    # Make sure we run in the script's folder to serve the correct files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)

    port = find_free_port(PORT)
    server_address = ('127.0.0.1', port)
    
    try:
        httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
    except Exception as e:
        print(f"Failed to start local server: {e}")
        sys.exit(1)

    print("==================================================")
    print("           NSQS HAND TRACKING SERVER              ")
    print("==================================================")
    print(f"Serving files from: {script_dir or os.getcwd()}")
    print(f"Local URL: http://localhost:{port}")
    print("==================================================")
    print("Press Ctrl+C to stop the server.")
    print("==================================================")

    # Automatically open default web browser
    def open_browser():
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        print("Server stopped. Goodbye!")

if __name__ == '__main__':
    main()