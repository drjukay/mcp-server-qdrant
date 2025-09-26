#!/usr/bin/env python3
"""
Wrapper script to run the MCP server with proper environment loading
"""
import os
import sys
from pathlib import Path

# Load environment from the main project
env_file = Path("/Users/jk/Library/CloudStorage/OneDrive-Personal/Dokumente/Github/FoM2526/.env")
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}", file=sys.stderr)
    except ImportError:
        # Fallback: manually parse .env file
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Set MCP-specific environment variables
os.environ['QDRANT_LOCAL_PATH'] = str(Path(__file__).parent / "../FoM2526/Folien+Lit/qdrant_db")
os.environ['COLLECTION_NAME'] = 'forschungsmethoden_literatur'
os.environ['EMBEDDING_PROVIDER'] = 'openai'
os.environ['EMBEDDING_MODEL'] = 'text-embedding-3-small'
os.environ['TOOL_FIND_DESCRIPTION'] = 'Suche nach relevanten Forschungsmethoden-Inhalten mit natürlicher Sprache. Verwende deutsche Begriffe für beste Ergebnisse.'
os.environ['TOOL_STORE_DESCRIPTION'] = 'Speichere wichtige Forschungsmethoden-Inhalte für spätere Verwendung. Der information Parameter sollte den Hauptinhalt enthalten.'

# Validate required environment
if not os.environ.get('OPENAI_API_KEY'):
    print("❌ Error: OPENAI_API_KEY not found in environment!", file=sys.stderr)
    sys.exit(1)

print(f"✅ OpenAI API key loaded: {os.environ.get('OPENAI_API_KEY', '')[:8]}...", file=sys.stderr)
print(f"✅ Qdrant path: {os.environ.get('QDRANT_LOCAL_PATH')}", file=sys.stderr)

# Import and run the main server
try:
    from mcp_server_qdrant.main import main
    main()
except Exception as e:
    print(f"❌ Error starting MCP server: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)