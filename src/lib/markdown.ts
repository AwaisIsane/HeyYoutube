// Tiny, safe markdown -> HTML renderer for summaries. Supports headings,
// unordered lists, bold, and paragraphs. Escapes all input first.

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function inline(s: string): string {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}

export function renderMarkdown(md: string): string {
  const lines = md.split('\n');
  const html: string[] = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push('</ul>');
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      closeList();
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (bullet) {
      if (!inList) {
        html.push('<ul>');
        inList = true;
      }
      html.push(`<li>${inline(bullet[1]!)}</li>`);
    } else if (heading) {
      closeList();
      const level = Math.min(4, Math.max(3, heading[1]!.length + 2));
      html.push(`<h${level}>${inline(heading[2]!)}</h${level}>`);
    } else {
      closeList();
      html.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return html.join('');
}
