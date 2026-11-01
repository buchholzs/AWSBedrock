all: build deploy invoke

build:
	agentcore build

deploy:
	agentcore deploy

invoke:
	agentcore invoke --runtime agent \
  --payload '{"prompt":"Liste alle meine heutigen Aufgaben auf."}' \

test:
	agentcore dev --mode browser

webgui:
	source agentcore/.env.local
	streamlit run streamlit/streamlit.py