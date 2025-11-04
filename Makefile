all: clean
	dotnet publish -c Release
	make cleanInt
	sh install_extensions.sh
clean: cleanSDK cleanArtifacts cleanInt
cleanSDK:
	dotnet clean
cleanArtifacts:
	rm -rf artifacts
	rm -rf Core-Model-v8-LINUX/build/Release Core-Model-v8-LINUX/build/Debug Core-model-v8/build/extensions
cleanInt:
	find . -type d -name bin -o -name obj | xargs rm -rf

