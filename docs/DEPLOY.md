# 배포 (CI/CD)

- **PR → main**: `.github/workflows/ci.yml`이 백엔드 테스트를 돌린다. 배포는 안 한다.
- **main에 push(머지)**: `.github/workflows/deploy.yml`이 테스트 → EC2 배포 → 배포 확인까지 한다.
- **수동 재배포**: GitHub Actions 탭 → Deploy → Run workflow. 또는 서버에서 `cd ~/tactilecheck && bash scripts/server_update.sh`.

## 지금 서버가 어떤 커밋인지

```bash
curl -s https://3-107-144-61.sslip.io/api/health
```

응답의 `commit`이 GitHub main의 최신 커밋 앞 7자리와 같으면 최신이다. `unknown`이면 커밋을 넣지 않고 빌드한 이미지다.

## 동작 방식

1. GitHub Actions가 배포 전용 SSH 키로 EC2에 접속한다.
2. 서버의 `~/.ssh/authorized_keys`가 이 키를 아래 명령 하나로만 묶어 둔다 (`restrict` + `command=`). 키가 새도 쉘·포트 포워딩은 쓸 수 없다.
   `cd ~/tactilecheck && git pull --ff-only && bash scripts/server_update.sh`
3. `server_update.sh`가 커밋 SHA를 넣어 Docker 이미지를 빌드하고 컨테이너를 바꾼다. 바꾸는 1~2분 동안은 요청이 실패할 수 있다.
4. Actions가 `/api/health`의 `commit`이 이번 커밋과 같은지 확인한다.

## 처음 한 번 설정 (저장소 관리자)

저장소 Secrets (Settings → Secrets and variables → Actions):

| 이름 | 값 |
| --- | --- |
| `DEPLOY_HOST` | EC2 주소 |
| `DEPLOY_USER` | SSH 사용자 (ubuntu) |
| `DEPLOY_SSH_KEY` | 배포 전용 개인 키 전체 |
| `DEPLOY_KNOWN_HOSTS` | `ssh-keyscan -t ed25519 <EC2 주소>` 결과 |

EC2 보안그룹은 22번 포트가 GitHub Actions에서 접근 가능해야 한다 (Actions IP는 고정이 아님). 비밀번호 로그인은 꺼 두고 키 로그인만 허용한다.
