all: build deploy invoke

build:
	agentcore build

deploy:
	agentcore deploy

invoke:
	agentcore invoke --runtime agent \
  --payload '{"prompt":"Liste alle meine Aufgaben auf."}' \

test:
	agentcore dev --mode browser