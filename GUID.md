2. Mariem : récupérer le projet GitHub
Sur son PC, elle ouvre VS Code puis un terminal et fait :
git clone https://github.com/BejaouiChaima/adaptive-recipe-copilot.git

Puis :
cd adaptive-recipe-copilot

Elle ouvre ensuite le projet dans VS Code :
code .

Elle doit avoir la même structure que toi :
adaptive-recipe-copilot/
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── api/
├── frontend/
├── tests/
├── configs/
├── .dvc/
├── .gitignore
├── requirements.txt
└── README.md

3. Installer l'environnement Python
Dans le terminal du projet :
python -m venv .venv

Puis :
.venv\Scripts\activate

Puis installer les dépendances du projet :
pip install -r requirements.txt

Et installer DVC avec Google Drive :
pip install dvc dvc-gdrive

4. Vérifier que le remote DVC est bien récupéré
Très important : normalement le remote que tu as configuré dans Git est déjà présent dans le projet.
Elle fait :
dvc remote list

Elle doit voir quelque chose comme :
gdrive    gdrive://...

Elle ne doit pas créer un nouveau remote.
5. Mariem crée SON projet Google Cloud
Elle va sur Google Cloud avec son propre compte Google.
Elle crée un nouveau projet, par exemple :
Adaptive Recipe Copilot - DVC - Mariem
Puis :
Activer Google Drive API
Google Cloud → APIs & Services → Library
Chercher :
Google Drive API
→ Enable.
6. Créer son OAuth consent screen
Dans Google Cloud :
Google Auth Platform
Elle configure :
- App name : DVC Desktop Mariem
- User type : External
- Publishing status : Testing
Puis elle ajoute son propre Gmail comme Test user.
7. Créer son propre OAuth Client
Dans son projet Google Cloud :
Google Auth Platform → Clients
→ Create client
Type :
Desktop app
Nom :
DVC Desktop Mariem

Elle obtient :
- Client ID
- Client Secret
👉 Ces informations restent chez Mariem.
Elle ne te les envoie pas.
8. Configurer DVC chez Mariem
Dans son terminal VS Code, dans le projet :
dvc remote modify gdrive gdrive_client_id "SON_CLIENT_ID"

Puis :
dvc remote modify --local gdrive gdrive_client_secret "SON_CLIENT_SECRET"

Le --local est important : son secret reste dans sa configuration locale et ne part pas sur GitHub.
9. Tester l'accès au même Google Drive
Elle fait :
dvc pull

Une fenêtre Google devrait s'ouvrir.
Elle se connecte avec son propre compte Google.
Google lui demandera l'autorisation d'accès.
Si tout est correctement configuré, DVC télécharge :
recipes.csv
reviews.csv

dans :
data/raw/

10. Iheb fait exactement la même chose
Iheb :
Son propre projet Google Cloud
Adaptive Recipe Copilot - DVC - Iheb

Son propre OAuth client
DVC Desktop Iheb

Son propre Client ID + Client Secret
Puis dans son projet cloné :
dvc remote modify gdrive gdrive_client_id "SON_CLIENT_ID"

et :
dvc remote modify --local gdrive gdrive_client_secret "SON_CLIENT_SECRET"

Puis :
dvc pull

Il s'authentifie avec son compte Google.