#!/bin/sh
ARTIFACTS_BASE="./artifacts/Release"
EXTENSIONS_PATH="$ARTIFACTS_BASE/extensions"
PUBLISH_PATH="$ARTIFACTS_BASE/publish"
PUBLISH_LINUX_PATH="$ARTIFACTS_BASE/linux-x64/publish"
mv "${PUBLISH_PATH}" "${EXTENSIONS_PATH}"
mv "${PUBLISH_LINUX_PATH}" "${PUBLISH_PATH}"
EXTENSION_ADMIN="${PUBLISH_PATH}/Landis.Extensions.dll"
cp extensions.xml "$EXTENSIONS_PATH"

echo dotnet "$EXTENSION_ADMIN" list
dotnet "$EXTENSION_ADMIN" list
dotnet "$EXTENSION_ADMIN" add "./Extension-Base-Fire/deploy/installer/Original Fire 5.txt"
dotnet "$EXTENSION_ADMIN" add "./Extension-Base-Wind/deploy/installer/Original Wind 4.txt"
dotnet "$EXTENSION_ADMIN" add "./Extension-Biomass-Succession/deploy/installer/Biomass Succession 7.txt"
dotnet "$EXTENSION_ADMIN" add "./Extension-Biomass-Harvest/deploy/installer/Biomass Harvest 6.txt"

cp -r "${EXTENSIONS_PATH}" "${PUBLISH_PATH}/"
cp Landis.Console.deps.json "${PUBLISH_PATH}/"
