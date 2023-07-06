import getpass
import json
from pathlib import Path
from sys import stdout
from urllib import request
from arachne import Arachne

if __name__ == "__main__":
    user = input("Benutzername: ")
    password = getpass.getpass("Passwort: ")
    print("Verbinde zum Server ...")
    db = Arachne(user, password, url="https://dienste.badw.de:9999/mlw", tbls=["zettel", "lemma"])
    print("Lade Daten herunter ...")
    zettels = db.zettel.search([{"c": "type", "o": "=", "v": 1}, {"c": "lemma_id", "o": ">", "v": 0}], select=["id", "lemma_id", "img_path", "u_date"])
    lemmata = db.lemma.getAll(select=["id", "lemma", "u_date"])
    lemmata_lemmata = list(map(lambda l: l["lemma"], lemmata))
    lemmata_ids = list(map(lambda l: l["id"], lemmata))
    print("Speichere Daten ...")
    Path("./zettel/").mkdir(exist_ok=True)
    zettels_downloaded = list(map(lambda p: int(p.stem), Path("./zettel/").iterdir()))
    zettels_to_download = []
    zettels_out = []
    last_modified = ""
    for l in lemmata:
        if last_modified < l["u_date"]: last_modified = l["u_date"]
    for z in zettels:
        if last_modified < z["u_date"]: last_modified = z["u_date"]
        if z["id"] not in zettels_downloaded: zettels_to_download.append({"id": z["id"], "img_path": z["img_path"]})
        zettel_lemma_id = lemmata_ids.index(z["lemma_id"])
        zettels_out.append({"id": z["id"], "lemma": lemmata_lemmata[zettel_lemma_id]})
    open("data.json", "w").write(json.dumps(zettels_out))
    print(f"Daten in 'data.json' gespeichert. Letzte Änderungen am {last_modified}")
    max_zettels_to_download = len(zettels_to_download)
    if max_zettels_to_download > 0:
        download_files = input(f"Sollen {max_zettels_to_download} Bilder heruntergeladen werden? (j/n) ")
        if download_files.lower() in ["j", "ja", "y", "yes"]:
            print("Bilder werden gesepichert ...")
            for i, z in enumerate(zettels_to_download):
                img = db.get_zettel_img(z["img_path"])
                open(Path(f"./zettel/{z['id']}.jpg"), "wb").write(img)
                stdout.write(f"\r{i+1}/{max_zettels_to_download}")
                stdout.flush()
            print("Alle Bilder gespeichert!")
    else: print("Keine Bilder zum Herunterladen!")
    db.close()
