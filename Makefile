all: clean
	dotnet build -c Release
	make cleanInt
clean: cleanSDK cleanInt
cleanSDK:
	dotnet clean
cleanInt:
	find . -type d -name bin -o -name obj | xargs rm -rf

