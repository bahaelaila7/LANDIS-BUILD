all: debug
release: clean build_release cleanInt
debug: clean build_debug cleanInt
clean: cleanSDK cleanArtifacts cleanInt
build_debug:
	dotnet publish -c Debug
	sh install_extensions.sh Debug
build_release:
	dotnet publish -c Release
	sh install_extensions.sh Release
cleanSDK:
	dotnet clean
cleanArtifacts:
	rm -rf artifacts
	rm -rf Core-Model-v8-LINUX/build/Release Core-Model-v8-LINUX/build/Debug Core-model-v8/build/extensions
cleanInt:
	find . -type d -name bin -o -name obj | xargs rm -rf

