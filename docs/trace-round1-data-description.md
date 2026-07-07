# Mo ta du lieu `trace-round1.jsonl` va `trace-round1.short.jsonl`

Tai lieu nay tong hop cau truc va dac diem cua hai file du lieu:

- `data/trace-round1.jsonl`: ban day du
- `data/trace-round1.short.jsonl`: ban cat gon

Muc tieu la giup nhin nhanh:

- schema cua moi record
- quy mo du lieu
- pattern hoi thoai multi-turn
- su khac nhau giua ban `full` va ban `short`
- nhung ket luan quan trong khi phan tich trace

## 1. Tong quan

Ca hai file deu o dinh dang JSON Lines (`.jsonl`):

- Moi dong la 1 JSON object hop le
- Tong cong `120` dong trong moi file
- Hai file co cung schema
- Khac nhau chu yeu o noi dung `body.messages[].content`

Kich thuoc xap xi:

- `trace-round1.jsonl`: khoang `14 MB`
- `trace-round1.short.jsonl`: khoang `57 KB`

## 2. Schema cua moi record

Moi dong co dang:

```json
{
  "request_id": 0,
  "timestamp_ms": 0,
  "workload_type": "conversation",
  "body": {
    "model": "Qwen3.5-2B",
    "messages": [
      { "role": "system", "content": "..." },
      { "role": "user", "content": "..." }
    ],
    "max_tokens": 200,
    "temperature": 0,
    "seed": 42
  }
}
```

Y nghia cac truong:

- `request_id`: ID cua request, tang tu `0` den `119`
- `timestamp_ms`: thoi diem den cua request theo millisecond
- `workload_type`: loai workload, trong trace nay luon la `conversation`
- `body.model`: model duoc benchmark, luon la `Qwen3.5-2B`
- `body.messages`: lich su hoi thoai
- `body.max_tokens`: gioi han token sinh ra, luon la `200`
- `body.temperature`: luon la `0`
- `body.seed`: luon la `42`

Trong `body.messages`, moi phan tu co:

- `role`: vai tro cua message, gom `system`, `user`, `assistant`
- `content`: noi dung text cua message

## 3. Gia tri co dinh va phan bo co ban

Nhung truong co dinh tren toan bo dataset:

- `workload_type = "conversation"`
- `body.model = "Qwen3.5-2B"`
- `body.max_tokens = 200`
- `body.temperature = 0`
- `body.seed = 42`

Phan bo do dai hoi thoai:

- `20` request co `2` message
- `20` request co `4` message
- `20` request co `6` message
- `20` request co `8` message
- `20` request co `10` message
- `20` request co `12` message

Tong so message trong ca file:

- `840` message

Pattern role trong hoi thoai:

- message dau tien luon la `system`
- sau do xen ke `user` va `assistant`
- cac record ngan nhat co dang: `system, user`
- cac record dai nhat co dang: `system, user, assistant, user, assistant, user, assistant, user, assistant, user, assistant, user`

## 4. Ban chat cau truc du lieu

Dataset nay khong phai `120` hoi thoai doc lap.

Thuc te no mo phong:

- `20` hoi thoai goc
- moi hoi thoai duoc luu o `6` moc do dai khac nhau
- cac moc la: `2, 4, 6, 8, 10, 12` message

Tong cong:

- `20 x 6 = 120` record

Co the hieu moi hoi thoai goc la 1 session user chat voi chatbot theo nhieu turn. Trong trace, BTC luu lai nhieu snapshot cua cung session do, moi snapshot dai hon snapshot truoc `2` message.

Vi du:

- dong `0` la session A o moc `2` message
- dong `20` la cung session A o moc `4` message
- dong `40` la cung session A o moc `6` message
- ...
- dong `100` la cung session A o moc `12` message

Quy luat nay dung cho toan bo `20` session.

## 5. Timestamp va tinh chat traffic

Hai truong nen quan sat:

- `request_id`: tang deu tu `0` den `119`
- `timestamp_ms`: tang don dieu tu `0` den `25475`

Dataset da sap xep theo thu tu thoi gian den. Khoang cach giua cac request khong phai luc nao cung bang nhau, nhung trong trace nay gia tri `timestamp_ms` duoc BTC co dinh san de mo phong traffic benchmark.

## 6. Khac nhau giua ban `full` va ban `short`

Hai file co:

- cung so dong
- cung schema
- cung `request_id`
- cung `timestamp_ms`
- cung `workload_type`
- cung `body.model`, `max_tokens`, `temperature`, `seed`
- cung so message trong tung record
- cung `role` cua tung message

Khac biet duy nhat nam o:

- `body.messages[].content`

Cu the:

- ban `full` giu noi dung day du cua moi message
- ban `short` chi giu `10 ky tu dau` cua moi `content`

Vi du:

```json
// full
{ "role": "user", "content": "node implementation capacity at batch stream rank, ..." }

// short
{ "role": "user", "content": "node imple" }
```

Toan bo `840` message trong file `short` deu tuan theo quy luat nay: `content_short` la prefix `10 ky tu` dau cua `content_full`.

## 7. Dac diem noi dung text

Noi dung trong `content` co dac diem:

- la tieng Anh synthetic
- mang dang "word-salad"
- lap lai nhieu tu khoa ky thuat
- khong phai hoi thoai tu nhien co nghia ro rang

Nhom tu xuat hien nhieu:

- `node`
- `batch`
- `stream`
- `gradient`
- `latency`
- `throughput`
- `kernel`
- `buffer`
- `benchmark`
- `deploy`
- `assistant`
- `query`
- `runtime`

Dieu nay phu hop voi boi canh benchmark serving LLM:

- du lieu duoc tao de mo phong tai production
- nhung khong phai du lieu user that
- uu tien tinh lap lai, tinh dong deu, va kha nang tai lap benchmark

## 8. Muc do trung lap cua content

`Content` trong ban `full` co trung lap, nhung la trung lap co chu dich do co che snapshot hoi thoai.

Nhan xet chinh:

- system prompt dau tien la giong nhau o ca `120` record
- cung 1 user message se xuat hien lai trong nhieu snapshot dai hon cua cung session
- cung 1 assistant message cung lap lai theo cach tuong tu

Vi vay:

- trung lap nay khong phai loi data
- ma la he qua truc tiep cua viec luu nhieu muc do dai cua cung 1 hoi thoai

## 9. Cach dung hai file trong phan tich

Nen dung `trace-round1.jsonl` khi:

- can phan tich noi dung thuc su
- can dem token, do dai prompt, prefix chung
- can nghien cuu caching, prefill, hay tinh chat multi-turn

Nen dung `trace-round1.short.jsonl` khi:

- can xem nhanh schema
- can debug parser
- can kiem tra thu tu request, role, timestamp
- can preview cau truc ma khong can doc text day du

## 10. Ket luan

Hai file `trace-round1.jsonl` va `trace-round1.short.jsonl` la hai phien ban cua cung mot trace benchmark:

- ban `full` dung de benchmark va phan tich noi dung that
- ban `short` la ban preview rut gon de de quan sat cau truc

Ve ban chat, dataset mo phong:

- `20` nguoi dung chat song song
- moi nguoi co toi da `6` luot user
- hoi thoai duoc luu thanh nhieu snapshot multi-turn
- noi dung la synthetic data, khong phai chat thuc te cua nguoi dung

Neu can trinh bay trace duoi dang de doc hon, co the tham khao them file:

- `docs/trace-round1-chat-simulation.md`
