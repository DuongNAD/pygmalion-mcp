# Thư viện chuyển động: nguồn và giấy phép

12 clip BVH trong thư mục này là thư viện offline của `animate(method="blender")`: `bpy_scripts/retarget.py` gắn chúng lên rig bất kỳ theo vai trò xương. Mọi file đều được phép phân phối lại trong repo công khai. 11 file lấy từ nguồn CC0 1.0. Riêng `bow.bvh` có thêm phần cúi chào viết cho repo này (xem bảng file). Không có dữ liệu Mixamo, không có dữ liệu non-commercial hay no-derivatives.

## Gói nguồn

Tải ngày 2026-10-23.

| Gói | Tác giả | Giấy phép | Trang | File đã tải |
|---|---|---|---|---|
| UAL1: Universal Animation Library [Standard] | Quaternius | CC0 1.0 (`License.txt` trong gói) | https://opengameart.org/node/174563 · https://quaternius.com/packs/universalanimationlibrary.html | https://opengameart.org/sites/default/files/universal_animation_librarystandard.zip, dùng `Godot/AnimationLibrary_Godot_Standard.glb` |
| UAL2: Universal Animation Library 2 [Standard] | Quaternius | CC0 1.0 (`License.txt` trong gói) | https://opengameart.org/content/universal-animation-library-2 | https://opengameart.org/sites/default/files/universal_animation_library_2standard.zip, dùng `Unreal-Godot/UAL2_Standard.glb` |
| KayKit: Adventurers Character Pack 1.0 | Kay Lousberg (kaylousberg.com) | CC0 1.0 (`LICENSE.txt` của repo) | https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0 | `addons/kaykit_character_pack_adventures/Characters/gltf/Knight.glb`, commit `aff755064be46dc287967fc5aff7fad64c898e62` |
| RobotExpressive (Animated Robot) | Tomás Laulhé (Quaternius); Don McCurdy chuyển sang glTF | CC0 1.0 (`README.md` cạnh file) | https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/RobotExpressive | `examples/models/gltf/RobotExpressive/RobotExpressive.glb`, commit `b924f0cad4058dc4dde71445c796980c3cd5b5ed` |

SHA-256 của file đã tải:
- `universal_animation_librarystandard.zip`: `18ff1a7215f4852b320203e8aaf02a1578b5c8eef9027fbaedfcedc7b85a3ac2`
- `universal_animation_library_2standard.zip`: `ec0e40d6d78fe9aaad59e322f40865a8675c22f0745e291622e54520391a9217`
- `Knight.glb`: `60428e3abc09ba83e595d256e3af8c5c976b46cdae599f0802fc82b4a3445168`
- `RobotExpressive.glb`: `047f5e5fb3bb6d378bd1df16ca6137f2a596c99b3a1b5690b4020c05aaf6f319`

## File

| File | Nguồn | Giây | Lặp | Giấy phép | Chỉnh sửa ngoài bước chuyển đổi chung |
|---|---|---|---|---|---|
| `idle_relaxed.bvh` | UAL1 `Idle_Loop` | 2,5 | có | CC0 1.0 | — |
| `idle_talking.bvh` | UAL1 `Idle_Talking_Loop` | 2,93 | có | CC0 1.0 | — |
| `idle_arms_crossed.bvh` | UAL2 `Idle_FoldArms_Loop` | 2,5 | có | CC0 1.0 | — |
| `walk_casual.bvh` | UAL1 `Walk_Loop` | 1,33 | có | CC0 1.0 | — |
| `jog.bvh` | UAL1 `Jog_Fwd_Loop` | 0,93 | có | CC0 1.0 | — |
| `jump_up.bvh` | KayKit Knight `Jump_Full_Short` | 1,17 | không | CC0 1.0 | — |
| `wave_hello.bvh` | UAL1 `Idle_Loop` + RobotExpressive `Wave` | 2,4 | không | CC0 1.0 | Toàn thân lấy từ `Idle_Loop`; xương đòn, cánh tay, cẳng tay và bàn tay phải lấy từ `Wave` (clip này chỉ key một phần cơ thể), quy về tư thế nghỉ của UAL1, trộn vào trong 0,2 s, giữ tư thế cuối của `Wave` rồi trộn ra trong 0,6 s |
| `dance.bvh` | UAL1 `Dance_Loop` | 1,0 | có | CC0 1.0 | — |
| `sit_down.bvh` | UAL1 `Sitting_Enter` | 1,8 | không | CC0 1.0 | Giữ tư thế cuối thêm 0,5 s |
| `punch.bvh` | UAL1 `Punch_Cross` | 1,0 | không | CC0 1.0 | — |
| `bow.bvh` | UAL1 `Idle_Loop` + cúi chào viết cho CharForge | 2,4 | không | CC0 1.0 (phần idle) + MIT của repo (phần cúi chào, xem `LICENSE`) | Thêm góc cúi theo trục ngang thế giới khi cúi hết (hông 15°, thân 25–38°, cổ 40°, đầu 42°, tay buông 10–16°, chân ngả 4°), hông lùi 7 cm, vào trong 0,25–0,95 s, giữ, về trong 1,55–2,3 s |
| `cheer.bvh` | KayKit Knight `Cheer` | 1,67 | không | CC0 1.0 | — |

## Bước chuyển đổi chung

Làm bằng Blender 4.5.4 LTS (importer glTF mặc định, cảnh 30 fps), một lần khi dựng thư viện:
- Khung xương chuẩn 22 khớp, tên kiểu Mixamo (`Hips`, `Spine`, `Spine1`, `Spine2`, `Neck`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`, `LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase`, và bên phải). BVH chuẩn: Y hướng lên, nhân vật nhìn +Z, đơn vị mét, 30 fps. Khớp gốc `Hips` nằm giữa hai khớp háng. Kênh gốc là `Xposition Yposition Zposition Zrotation Xrotation Yrotation`, các khớp khác là `Zrotation Xrotation Yrotation`.
- Tư thế nghỉ (OFFSET) là tư thế nghỉ của rig nguồn: T-pose cho UAL1, UAL2 và KayKit. Mỗi khớp mang đúng phép xoay thế giới của xương nguồn tương ứng so với tư thế nghỉ của xương đó, nên mọi khung hình tái tạo đúng hướng từng đoạn xương nguồn.
- Vai trò mà rig nguồn không có thì đi theo khớp cha, không tự xoay. KayKit không có cổ, ngực trên và xương đòn. RobotExpressive không có ngón chân.
- Ngón tay và các xương IK hay điều khiển bị bỏ.
- Kiểm tra: mỗi file được nhập lại bằng importer BVH của Blender và so với dữ liệu đã ghi. Sai số xoay là 0°, sai số vị trí khớp ≤ 0,2 mm.

Thêm clip mới: đặt file `.bvh`, `.glb` hoặc `.fbx` (khung xương humanoid đứng thẳng) vào đây hoặc vào project, khai báo trong `config.MOTION_LIBRARY` hoặc trong `project.json` → `motion_library`, rồi ghi nguồn và giấy phép vào bảng trên.
