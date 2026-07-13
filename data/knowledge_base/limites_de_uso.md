# Limites de Uso por Plano — Aurora Cloud

| Plano | Armazenamento | Chamadas de API/mês | Usuários |
|---|---|---|---|
| Starter | 10 GB | 5.000 | 1 |
| Pro | 100 GB | 100.000 | 5 |
| Business | 1 TB | 1.000.000 | 25 |
| Enterprise | Sob consulta | Sob consulta | Ilimitado |

Ao ultrapassar o limite de chamadas de API do plano, requisições adicionais
são cobradas a **US$ 0,002 por chamada** no plano Pro e Business. O plano
Starter não permite excedente: requisições acima do limite retornam erro
`429 Too Many Requests` até o próximo ciclo de faturamento.

Upgrades de plano têm efeito imediato. Downgrades entram em vigor apenas
no próximo ciclo de faturamento, para evitar perda de dados por excesso
de armazenamento.
