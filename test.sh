#!/bin/sh
rm -rf ./data
.venv/bin/mailmind init --root ./data
.venv/bin/mailmind index-mbox --mbox sample.mbox --account demo
.venv/bin/mailmind worker attachments --max 100 --langs eng+spa+swe 
.venv/bin/mailmind worker embeddings --max 500
