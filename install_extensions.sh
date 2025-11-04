#!/bin/sh
ARTIFACTS_BASE="./artifacts/Release"
EXTENSIONS_PATH="$ARTIFACTS_BASE/extensions"
PUBLISH_PATH="$ARTIFACTS_BASE/publish"
PUBLISH_LINUX_PATH="$ARTIFACTS_BASE/linux-x64/publish"
mv "${PUBLISH_PATH}" "${EXTENSIONS_PATH}"
mv "${PUBLISH_LINUX_PATH}" "${PUBLISH_PATH}"
EXTENSION_ADMIN="${PUBLISH_PATH}/Landis.Extensions.dll"
DEPS_PATH="${PUBLISH_PATH}/Landis.Console.deps.json"
cp extensions.xml "$EXTENSIONS_PATH"
cp -r "${EXTENSIONS_PATH}" "${PUBLISH_PATH}/"
EXTENSIONS_PATH="$PUBLISH_PATH/extensions"
#dotnet "$EXTENSION_ADMIN" add "./Extension-Base-Fire/deploy/installer/Original Fire 5.txt"
#./add_extension_to_deps.py --json-path "${DEPS_PATH}" --extension-dll "${EXTENSIONS_PATH}/Landis.Extension.OriginalFire-v5.dll"
#dotnet "$EXTENSION_ADMIN" add "./Extension-Base-Wind/deploy/installer/Original Wind 4.txt"
#./add_extension_to_deps.py --json-path "${DEPS_PATH}" --extension-dll "${EXTENSIONS_PATH}/Landis.Extension.OriginalWind-v4.dll"
#dotnet "$EXTENSION_ADMIN" add "./Extension-Biomass-Succession/deploy/installer/Biomass Succession 7.txt"
#./add_extension_to_deps.py --json-path "${DEPS_PATH}" --extension-dll "${EXTENSIONS_PATH}/Landis.Extension.Succession.Biomass-v7.dll"
#dotnet "$EXTENSION_ADMIN" add "./Extension-Biomass-Harvest/deploy/installer/Biomass Harvest 6.txt"
#./add_extension_to_deps.py --json-path "${DEPS_PATH}" --extension-dll "${EXTENSIONS_PATH}/Landis.Extension.BiomassHarvest-v6.dll"
#dotnet "$EXTENSION_ADMIN" add "./Extension-Output-Biomass/deploy/installer/Output Biomass 4.txt"
#./add_extension_to_deps.py --json-path "${DEPS_PATH}" --extension-dll "${EXTENSIONS_PATH}/Landis.Extension.Output.Biomass-v4.dll"

EXTENSIONS_TXT=(
Extension-Base-BDA/deploy/installer/'Climate BDA 5.txt'
Extension-Base-EDA/deploy/installer/'EDA 3.txt'
Extension-Base-Fire/deploy/installer/'Original Fire 5.txt'
Extension-Base-Wind/deploy/installer/'Original Wind 4.txt'
Extension-Biomass-Harvest/deploy/installer/'Biomass Harvest 6.txt'
Extension-Biomass-Succession/deploy/installer/'Biomass Succession 7.txt'
Extension-Dynamic-Biomass-Fuels/deploy/installer/'Dynamic Fuels 4.txt'
Extension-Dynamic-Fire-System/deploy/installer/'Dynamic Fire Component 4.txt'
Extension-ForCS-Succession/deploy/installer/'ForCS 4.0.2.txt'
Extension-Land-Use-Plus/deploy/installer/'Land Use 4.txt'
Extension-LinearWind/deploy/installer/'Linear Wind 3.txt'
Extension-Local-Habitat-Suitability-Output/deploy/installer/'Local Habitat Output.txt'
Extension-NECN-Succession/deploy/installer/NECN_Succession8.txt
Extension-Output-Biomass-Community/deploy/installer/'Output Biomass Community 3.txt'
Extension-Output-Biomass/deploy/installer/'Output Biomass 4.txt'
Extension-Output-Wildlife-Habitat/deploy/installer/'Wildlife Habitat Output 3.txt'
Extension-Social-Climate-Fire/deploy/installer/'Scrapple 4.txt'
Extension-SOSIEL-Harvest/deploy/installer/'SHE 2.txt'
)
for i in "${EXTENSIONS_TXT[@]}"; do
    dotnet "$EXTENSION_ADMIN" add "$i"
done;
for i in $(ls "$EXTENSIONS_PATH/Landis.Extension."*.dll); do
    ./add_extension_to_deps.py --json-path "${DEPS_PATH}" --extension-dll "$i"
done;

dotnet "$EXTENSION_ADMIN" list
