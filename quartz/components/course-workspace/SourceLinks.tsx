interface SourceLinksProps {
  pdf?: string
  reader?: string
  video?: string
}

export function WorkspaceSourceLinks({ pdf, reader, video }: SourceLinksProps) {
  return (
    <div class="course-source-links">
      {reader && (
        <a
          href={`${reader}?view=reader`}
          target="_blank"
          rel="noopener noreferrer"
          data-router-ignore
        >
          open reader ↗
        </a>
      )}
      {pdf && (
        <a href={`/${pdf}?raw=1`} target="_blank" rel="noopener noreferrer">
          pdf ↗
        </a>
      )}
      {video && (
        <a
          href={`https://www.youtube.com/watch?v=${encodeURIComponent(video)}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          video ↗
        </a>
      )}
      <span class="course-source-license">mit ocw · cc by-nc-sa</span>
    </div>
  )
}
